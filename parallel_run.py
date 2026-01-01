import csv
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

# --------- CONFIGURE CPU CORES ----------
# Use all available cores (or set a fixed number like 4)
set_num_threads(get_num_threads())

# --------- PARALLEL BITONIC SORT ----------
@njit(parallel=True)
def bitonic_sort_parallel(arr):
    n = arr.shape[0]
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in prange(n):
                ixj = i ^ j
                if ixj > i:
                    if ((i & k) == 0 and arr[i] > arr[ixj]) or \
                       ((i & k) != 0 and arr[i] < arr[ixj]):
                        arr[i], arr[ixj] = arr[ixj], arr[i]
            j //= 2
        k *= 2

# --------- HELPER ----------
def next_power_of_two(n):
    return 1 << (n - 1).bit_length()

# --------- READ CSV ----------
numbers = []
with open("dataset.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    for row in reader:
        if row:
            numbers.append(int(row[0]))

original_len = len(numbers)

# --------- PAD TO POWER OF 2 ----------
target_len = next_power_of_two(original_len)
pad_value = max(numbers)
numbers.extend([pad_value] * (target_len - original_len))

# --------- CONVERT TO NUMPY ----------
arr = np.array(numbers, dtype=np.int32)

# --------- RUN PARALLEL BITONIC SORT ----------
bitonic_sort_parallel(arr)

# --------- REMOVE PADDING ----------
sorted_numbers = arr[:original_len].tolist()


# ---------- WRITE SORTED DATA TO NEW CSV ----------
with open("sorted_dataset_parallel.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write header (same as input)
    writer.writerow(["values"])

    # Write sorted integers
    for value in sorted_numbers:
        writer.writerow([value])

print("Sorted dataset (parallely) written to sorted_dataset.csv")
