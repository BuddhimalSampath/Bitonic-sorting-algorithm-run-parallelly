import numpy as np
import pandas as pd
from mpi4py import MPI
import math
import sys

# ===================== UTILITY FUNCTIONS =====================

def compare_and_swap(a, b, direction):
    """Compare two values and swap according to direction."""
    if (direction == 1 and a > b) or (direction == 0 and a < b):
        return b, a
    return a, b

def bitonic_sort_local(arr, up=True):
    """Serial bitonic sort on local array."""
    n = len(arr)
    if n <= 1:
        return arr
    k = n // 2
    first = bitonic_sort_local(arr[:k], True)
    second = bitonic_sort_local(arr[k:], False)
    combined = np.concatenate([first, second])
    return bitonic_merge_local(combined, up)

def bitonic_merge_local(arr, up=True):
    """Serial bitonic merge on local array."""
    n = len(arr)
    if n <= 1:
        return arr
    k = n // 2
    for i in range(k):
        arr[i], arr[i+k] = compare_and_swap(arr[i], arr[i+k], up)
    first = bitonic_merge_local(arr[:k], up)
    second = bitonic_merge_local(arr[k:], up)
    return np.concatenate([first, second])

# ===================== PARALLEL BITONIC SORT =====================

def parallel_bitonic_sort(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast total number of elements (N)
    n = len(data) if rank == 0 else None
    n = comm.bcast(n, root=0)

    # Calculate local chunk size
    local_n = n // size
    local_data = np.zeros(local_n, dtype=np.int64)

    # Scatter data to all processes
    # NOTE: Since N and Size are powers of 2, N is always divisible by Size.
    comm.Scatter(data, local_data, root=0)

    # Step 1: Local bitonic sort
    # Start by sorting the local chunk ascending
    local_data = bitonic_sort_local(local_data, up=True)

    # Step 2: Parallel bitonic merge across processes
    # We iterate through the dimensions of the hypercube
    num_stages = int(math.log2(size))
    
    for stage in range(1, num_stages + 1):
        partner_distance = 2 ** (stage - 1)
        partner = rank ^ partner_distance  # XOR to find partner in hypercube

        # Exchange data with partner
        recv_data = np.zeros(local_n, dtype=np.int64)
        comm.Sendrecv(local_data, dest=partner, sendtag=0,
                      recvbuf=recv_data, source=partner, recvtag=0)

        # Decide merge direction based on Bitonic logic
        # If the virtual group index is even, we want ascending; else descending.
        if ((rank // (2**stage)) % 2) == 0:
            direction = True # Ascending (keep min) implies we look at rank vs partner
        else:
            direction = False # Descending

        # Determine if I keep smaller or larger half
        # Logic: If I am the lower rank in the pair and direction is Up -> I keep small
        #        If I am the higher rank in the pair and direction is Up -> I keep large
        combined = np.concatenate([local_data, recv_data])
        combined_sorted = np.sort(combined)

        if direction == True:
            # Sort Ascending across the pair
            if rank < partner:
                local_data = combined_sorted[:local_n] # Keep Small
            else:
                local_data = combined_sorted[-local_n:] # Keep Large
        else:
            # Sort Descending across the pair
            if rank < partner:
                local_data = combined_sorted[-local_n:] # Keep Large
            else:
                local_data = combined_sorted[:local_n] # Keep Small

    # Gather fully sorted array at root
    sorted_data = None
    if rank == 0:
        sorted_data = np.zeros(n, dtype=np.int64)
    
    comm.Gather(local_data, sorted_data, root=0)

    return sorted_data

# ===================== HELPER FUNCTIONS =====================

def pad_to_power_of_two(arr):
    """Pad array to nearest power of 2."""
    n = len(arr)
    if (n & (n - 1)) == 0:
        return arr, n
    next_pow = 2**int(np.ceil(np.log2(n)))
    # Pad with max value + 1 so they naturally float to the end/top during sort
    padded = np.concatenate([arr, np.full(next_pow - n, np.max(arr)+1)])
    return padded, n

# ===================== MAIN =====================

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # CRITICAL CHECK: Size must be power of 2
    if (size & (size - 1)) != 0:
        if rank == 0:
            # Print specifically to stderr or stdout so app.py can catch it
            print(f"ERROR: Process count {size} is not a power of 2. Exiting.")
        sys.exit(1)

    # Rank 0 loads the data
    if rank == 0:
        try:
            df = pd.read_csv("dataset_long.csv")
            col = df.columns[0]
            data = df[col].values
            data, original_size = pad_to_power_of_two(data)
            n = len(data)
            # print(f"DEBUG: Padded size: {n}")
        except Exception as e:
            print(f"ERROR: Could not read dataset. {e}")
            sys.exit(1)
    else:
        data = None
        n = None
        original_size = None

    # Broadcast original size for trimming later
    original_size = comm.bcast(original_size if rank==0 else None, root=0)

    # Synchronize before timing
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Run Sort
    sorted_data = parallel_bitonic_sort(data, comm)
    
    # ... existing code ...

    comm.Barrier()
    end_time = MPI.Wtime()

    # Root saves data and prints results
    if rank == 0:
        # 1. Remove the padding (restore original size)
        final_sorted_data = sorted_data[:original_size]
        
        # 2. Save to CSV
        # We use Pandas to write the file quickly
        output_filename = "dataset_sorted.csv"
        pd.DataFrame(final_sorted_data, columns=["SortedValues"]).to_csv(output_filename, index=False)
        
        # 3. Print the metric for the dashboard
        # IMPORTANT: This must remain the LAST print statement so app.py can read it.
        print(f"{size},{end_time - start_time}")

if __name__ == "__main__":
    main()