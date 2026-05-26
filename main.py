import os
import sys
import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Tuple, Iterator
from huggingface_hub import HfApi, list_repo_tree, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm
from algorithms import Tokenizer

# ==========================================================
# WORKER IDENTIFICATION
# ==========================================================
# This value is updated by repo_clone.py during repository creation
WORKER_ID: int = 9

# ==========================================================
# CONFIGURATION LOAD
# ==========================================================
def load_config() -> dict:
    """Loads configuration settings from local metadata.json file."""
    config_path = "metadata.json"
    if not os.path.exists(config_path):
        return {
            "chunk_size": 2500000,
            "active_workers": 20,
            "hf_repo_id": "anisoleai/fineweb-tokenized",
            "hf_dataset": "HuggingFaceFW/fineweb",
            "github_org": "LeadbaseAI-Official"
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================================
# PROGRESS MANAGEMENT ON HF HUB
# ==========================================================
def load_progress(
    api: HfApi,
    repo_id: str,
    worker_id: int,
    token: Optional[str]
) -> Tuple[int, int]:
    """
    Downloads and reads the metadata.json for this worker from the Hugging Face repo.
    Returns a tuple of (processed_docs, shard_index).
    """
    filename = f"data_{worker_id}/metadata.json"
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("processed_docs", 0)), int(data.get("shard_index", 0))
    except EntryNotFoundError:
        tqdm.write(f"No existing metadata.json found at {filename}. Starting from scratch.")
        return 0, 0
    except Exception as e:
        tqdm.write(f"Warning loading progress from HF: {e}. Fallback to (0, 0).")
        return 0, 0

def save_progress(
    api: HfApi,
    repo_id: str,
    worker_id: int,
    token: Optional[str],
    processed_docs: int,
    shard_index: int
) -> None:
    """
    Saves and uploads the updated progress metadata.json to the Hugging Face repo.
    """
    filename = f"data_{worker_id}/metadata.json"
    local_path = "metadata_worker.json"
    progress = {
        "processed_docs": processed_docs,
        "shard_index": shard_index
    }
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    tqdm.write(f"Uploaded updated progress metadata to HF: {filename} -> {progress}")

# ==========================================================
# REMOTE PARQUET STREAMER
# ==========================================================
def get_parquet_files(api: HfApi, dataset_id: str, token: Optional[str]) -> List[dict]:
    """
    List sorted parquet file paths and their row counts from the FineWeb dataset repository.
    Caches the file paths and row counts locally in parquet_files.json to avoid rate limits.
    """
    cache_path = "parquet_files.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                paths = json.load(f)
                if paths:
                    tqdm.write(f"Loaded {len(paths):,} Parquet file paths from local cache.")
                    return paths
        except Exception as e:
            tqdm.write(f"Warning loading cached parquet files list: {e}")

    tqdm.write("Listing dump directories...")
    try:
        top_entries = list(list_repo_tree(
            repo_id=dataset_id,
            path_in_repo="data",
            repo_type="dataset",
            token=token,
            recursive=False,
        ))
    except Exception as e:
        tqdm.write(f"Error listing top directory: {e}")
        return []

    dump_dirs: List[str] = sorted([
        entry.path for entry in top_entries
        if hasattr(entry, 'path') and entry.path.startswith("data/CC-MAIN-")
    ])

    parquet_files: List[dict] = []
    for dump_dir in tqdm(dump_dirs, desc="Scanning dumps for Parquet files"):
        try:
            entries = list(list_repo_tree(
                repo_id=dataset_id,
                path_in_repo=dump_dir,
                repo_type="dataset",
                token=token,
                recursive=False,
            ))
            paths = sorted([
                entry.path for entry in entries
                if hasattr(entry, 'path') and entry.path.endswith(".parquet")
            ])
            for p in paths:
                hf_path = f"hf://datasets/{dataset_id}/{p}"
                try:
                    pf = pq.ParquetFile(hf_path)
                    parquet_files.append({"path": p, "num_rows": pf.metadata.num_rows})
                except Exception:
                    continue
        except Exception as e:
            tqdm.write(f"Warning: could not list files in {dump_dir}: {e}")
            continue

    if parquet_files:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(parquet_files, f, indent=2)
            tqdm.write(f"Cached {len(parquet_files):,} Parquet file paths and row counts to {cache_path}.")
        except Exception as e:
            tqdm.write(f"Warning caching parquet files list: {e}")

    return parquet_files


def stream_fineweb_rows(
    dataset_id: str,
    parquet_files: List[dict],
    start_row: int,
    end_limit: int,
    token: Optional[str]
) -> Iterator[Tuple[str, int]]:
    """
    Generator streaming 'text' rows remotely from FineWeb parquet files.
    Only downloads relevant column bytes using HTTP range requests via pyarrow.
    Yields (text_string, global_row_index).
    """
    # Disable HF transfer acceleration features for remote parquet range requests.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"

    global_offset = 0

    for file_info in parquet_files:
        pq_path = file_info["path"]
        num_rows = file_info["num_rows"]

        if global_offset >= end_limit:
            break

        if global_offset + num_rows <= start_row:
            global_offset += num_rows
            continue

        hf_path = f"hf://datasets/{dataset_id}/{pq_path}"
        tqdm.write(f"Streaming remote shard {pq_path} ({num_rows:,} rows)...")

        try:
            pf = pq.ParquetFile(hf_path)
            num_row_groups = pf.num_row_groups
        except Exception as e:
            tqdm.write(f"Warning: could not read metadata for {pq_path}: {e}")
            continue

        for rg_idx in range(num_row_groups):
            if global_offset >= end_limit:
                break

            try:
                rg_meta = pf.metadata.row_group(rg_idx)
                rg_rows = rg_meta.num_rows
            except Exception as e:
                tqdm.write(f"Warning: could not read metadata for row group {rg_idx} of {pq_path}: {e}")
                continue

            if global_offset + rg_rows <= start_row:
                global_offset += rg_rows
                continue

            try:
                table = pf.read_row_group(rg_idx, columns=["text"], use_threads=False)
                texts = table.column("text").to_pylist()
            except Exception as e:
                tqdm.write(f"Warning: could not read row group {rg_idx} of {pq_path}: {e}")
                global_offset += rg_rows
                continue

            rg_start = max(0, start_row - global_offset)
            rg_end = min(rg_rows, end_limit - global_offset)

            for i in range(rg_start, rg_end):
                text_val = texts[i]
                if text_val is not None:
                    yield str(text_val), global_offset + i

            global_offset += rg_rows

# ==========================================================
# SHARD PROCESSING LOOP
# ==========================================================
def process_single_shard(
    api: HfApi,
    hf_dataset: str,
    hf_repo_id: str,
    hf_token: str,
    start_row: int,
    end_limit: int,
    shard_index: int,
    processed_docs: int,
    tokenizer: Tokenizer,
    eos_token_id: int
) -> bool:
    """
    Downloads, tokenizes and uploads a single shard.
    Returns True if the shard was processed successfully and we should continue,
    False if we ran out of documents.
    """
    tqdm.write(f"\n--- Starting Shard {shard_index} ---")
    tqdm.write(f"Orchestration Range: Row {start_row:,} to Row {end_limit:,}")

    schema = pa.schema([
        ('token_ids', pa.uint16())
    ])
    temp_parquet_path = f"temp_{shard_index}.parquet"
    writer = pq.ParquetWriter(temp_parquet_path, schema, compression='SNAPPY')

    total_tokens_written: int = 0
    processed_docs_in_run: int = 0
    token_limit: int = 2000000000  # 2.0 Billion Token Limit (occupies ~4 GB of uint16 data)
    flush_buffer: List[int] = []
    flush_threshold: int = 50000000  # Flush to disk every 50M tokens
    has_data: bool = False
    batch: List[str] = []
    batch_size: int = 20000  # Tokenize in batches of 20,000 docs in parallel

    # List parquet files before starting the progress bar so we don't interrupt it!
    parquet_files = get_parquet_files(api, hf_dataset, hf_token)
    if not parquet_files:
        tqdm.write("No Parquet files found to process.")
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)
        return False

    row_stream = stream_fineweb_rows(
        dataset_id=hf_dataset,
        parquet_files=parquet_files,
        start_row=start_row,
        end_limit=end_limit,
        token=hf_token
    )

    try:
        pbar = tqdm(total=token_limit, desc=f"Tokenizing Shard {shard_index}", unit="tokens")
        for text, _ in row_stream:
            has_data = True
            processed_docs_in_run += 1

            batch.append(text)

            if len(batch) >= batch_size:
                # Tokenize in parallel using Rust BPE batch encoding
                encodings = tokenizer.tokenizer.encode_batch(batch)
                tokens_added: int = 0
                for enc in encodings:
                    flush_buffer.extend(enc.ids)
                    flush_buffer.append(eos_token_id)
                    tokens_added += len(enc.ids) + 1
                
                pbar.update(tokens_added)
                batch.clear()

                if len(flush_buffer) >= flush_threshold:
                    arr = np.array(flush_buffer, dtype=np.uint16)
                    table = pa.Table.from_arrays([pa.array(arr, type=pa.uint16())], names=['token_ids'])
                    writer.write_table(table)
                    total_tokens_written += len(flush_buffer)
                    flush_buffer.clear()

                if (total_tokens_written + len(flush_buffer)) >= token_limit:
                    break
        
        # Process remaining items in batch
        if batch and (total_tokens_written + len(flush_buffer)) < token_limit:
            encodings = tokenizer.tokenizer.encode_batch(batch)
            tokens_added = 0
            for enc in encodings:
                flush_buffer.extend(enc.ids)
                flush_buffer.append(eos_token_id)
                tokens_added += len(enc.ids) + 1
            pbar.update(tokens_added)
            batch.clear()
            
        pbar.close()

        if flush_buffer:
            arr = np.array(flush_buffer, dtype=np.uint16)
            table = pa.Table.from_arrays([pa.array(arr, type=pa.uint16())], names=['token_ids'])
            writer.write_table(table)
            total_tokens_written += len(flush_buffer)
            flush_buffer.clear()

    finally:
        writer.close()

    if total_tokens_written == 0:
        tqdm.write("No new tokens generated in this range.")
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)
        return has_data

    shard_name = f"data_{WORKER_ID}/shard-{shard_index:05d}.parquet"
    tqdm.write(f"Uploading shard to HF: {shard_name} ({total_tokens_written:,} tokens)...")
    
    try:
        api.upload_file(
            path_or_fileobj=temp_parquet_path,
            path_in_repo=shard_name,
            repo_id=hf_repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        tqdm.write("Parquet shard uploaded successfully.")
        
        save_progress(
            api=api,
            repo_id=hf_repo_id,
            worker_id=WORKER_ID,
            token=hf_token,
            processed_docs=processed_docs + processed_docs_in_run,
            shard_index=shard_index + 1
        )
        tqdm.write("Progress metadata updated successfully.")
    except Exception as e:
        tqdm.write(f"Error during HuggingFace upload/update: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)

    return True

# ==========================================================
# SELF-RESTART TRIGGER (GITHUB WORKFLOW DISPATCH)
# ==========================================================
def trigger_self_restart() -> None:
    """Triggers a workflow dispatch on GitHub to restart this runner before timeout."""
    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    ref = os.getenv("GITHUB_REF_NAME", "main")
    
    if not repo or not token:
        print("Warning: GITHUB_REPOSITORY or GITHUB_TOKEN env variables are missing. Cannot self-restart.")
        return
        
    url = f"https://api.github.com/repos/{repo}/actions/workflows/main.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "ref": ref
    }
    
    try:
        import requests
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 204:
            print("Successfully triggered self-restart workflow dispatch.")
        else:
            print(f"Failed to trigger self-restart workflow: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Error triggering self-restart workflow: {e}")

# ==========================================================
# MAIN INFINITE LOOP WITH TIME CAP
# ==========================================================
def main() -> None:
    print("=== FineWeb Worker Tokenization Pipeline (Infinite Grid-Strided) ===")
    start_time = time.time()
    max_duration = 14400  # 4 Hours in seconds

    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi()
    tokenizer = Tokenizer("tokenizer.json")
    eos_token_id: int = 52002
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}, EOS Token ID: {eos_token_id}")

    while True:
        # Load fresh config and progress at the start of each shard iteration
        config = load_config()
        chunk_size: int = int(config["chunk_size"])
        active_workers: int = int(config["active_workers"])
        hf_repo_id: str = config["hf_repo_id"]
        hf_dataset: str = config["hf_dataset"]

        processed_docs, shard_index = load_progress(api, hf_repo_id, WORKER_ID, hf_token)
        
        # Calculate grid indexing
        start_row: int = ((WORKER_ID - 1) + shard_index * active_workers) * chunk_size
        end_limit: int = start_row + chunk_size

        print(f"\n=============================================")
        print(f"Worker ID: {WORKER_ID} | Shard Index: {shard_index}")
        print(f"Processed Documents: {processed_docs:,}")
        print(f"=============================================")

        # Run the shard
        more_data = process_single_shard(
            api=api,
            hf_dataset=hf_dataset,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            start_row=start_row,
            end_limit=end_limit,
            shard_index=shard_index,
            processed_docs=processed_docs,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id
        )

        if not more_data:
            print("\nReached the end of the FineWeb dataset. Exiting loop.")
            break

        # Check running duration
        elapsed_time = time.time() - start_time
        print(f"Elapsed running time: {elapsed_time / 3600:.2f} hours.")
        if elapsed_time >= max_duration:
            print(f"\nRunning time has exceeded the 4-hour limit ({elapsed_time / 3600:.2f}h).")
            print("Triggering self-restart on GitHub Actions...")
            trigger_self_restart()
            break

        print("\nShard completed. Proceeding to the next shard...")

if __name__ == "__main__":
    main()
