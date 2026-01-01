import csv
import math

# ---------- Bitonic Sort Implementation ----------

def bitonic_compare(arr, i, j, direction):
    if (direction == 1 and arr[i] > arr[j]) or (direction == 0 and arr[i] < arr[j]):
        arr[i], arr[j] = arr[j], arr[i]

def bitonic_merge(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            bitonic_compare(arr, i, i + k, direction)
        bitonic_merge(arr, low, k, direction)
        bitonic_merge(arr, low + k, k, direction)

def bitonic_sort_recursive(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        bitonic_sort_recursive(arr, low, k, 1)   # ascending
        bitonic_sort_recursive(arr, low + k, k, 0)  # descending
        bitonic_merge(arr, low, cnt, direction)

def bitonic_sort(arr, direction=1):
    bitonic_sort_recursive(arr, 0, len(arr), direction)


# ---------- Helper Functions ----------

def next_power_of_two(n):
    return 1 << (n - 1).bit_length()

# ---------- Read CSV File ----------

numbers = []

with open("dataset.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # skip column name
    for row in reader:
        if row:
            numbers.append(int(row[0]))

print("Original Data:", numbers)

# ---------- Pad to Power of 2 ----------

original_length = len(numbers)
target_length = next_power_of_two(original_length)

if target_length != original_length:
    max_value = max(numbers)
    numbers.extend([max_value] * (target_length - original_length))

# ---------- Sort Using Bitonic Sort ----------

bitonic_sort(numbers, direction=1)

# Remove padding
sorted_numbers = numbers[:original_length]
print("")
print("Sorted Data:", sorted_numbers)
