import random as rd
import time
import threading
import math
# Sort Algorithms
def name_to_function_dict():
    """
    To be updated whenever there are changes in the ...stats.csv files!
    This maps names into function name conventions for calling.
    """
    return {
        "Linear Search": "linear_search",
        "Binary Search": "binary_search",
        "Jump Search": "jump_search",
        "Exponential Search": "exponential_search",
        "Interpolation Search": "interpolation_search",
        "Fibonacci Search": "fibonacci_search",
        "Ternary Search": "ternary_search",
        "Binary Search Tree": "binary_search_tree",
        "Bidirectional Search": "bidirectional_search",
        "Jump Point Search": "jump_point_search",
        "Genetic Algorithm Search": "genetic_algorithm_search",
        "Simulated Annealing": "simulated_annealing",
        "Bubble Sort": "bubble_sort",
        "Insertion Sort": "insertion_sort",
        "Selection Sort": "selection_sort",
        "Merge Sort": "merge_sort",
        "Quick Sort": "quick_sort",
        "Heap Sort": "heap_sort",
        "Counting Sort": "counting_sort",
        "Radix Sort": "radix_sort",
        "Shell Sort": "shell_sort",
        "Cocktail Shaker Sort": "cocktail_shaker_sort",
        "Gnome Sort": "gnome_sort",
        "Pancake Sort": "pancake_sort",
        "Bogo Sort": "bogo_sort",
        "Comb Sort": "comb_sort",
        "Tim Sort": "tim_sort",
        "Sleep Sort": "sleep_sort",
        "Stooge Sort": "stooge_sort",
        "Bitonic Sort": "bitonic_sort",
        "Odd-Even Sort": "odd_even_sort",
        "Pigeonhole Sort": "pigeonhole_sort",
        "Smooth Sort": "smooth_sort"
    }


def bubble_sort(problem: list) -> list:
    for length in range(len(problem)):
        for i in range(len(problem)-1):
            if problem[i] > problem[i+1]:
                problem[i], problem[i+1] = problem[i+1], problem[i]


def insertion_sort(problem: list) -> list:
    for i, el in enumerate(problem):
        comparitor = el
        j = i - 1
        while j > 0 and comparitor < problem[j]:
            problem[j+1] = problem[j]
            j = j - 1
        problem[j+1] = comparitor


def selection_sort(problem: list) -> list:
    n = len(problem)
    for selector in range(n):
        min_index = selector
        for i in range(selector + 1, n):
            if problem[i] < problem[min_index]:
                min_index = i
        # Swap the found minimum element with the first element
        problem[selector], problem[min_index] = problem[min_index], problem[selector]


def merge_sort(problem: list) -> list:
    if len(problem) <= 1:
        return problem
    else:
        left, right = problem[:len(problem)//2], problem[len(problem)//2:]
        sorted_problem = []
        while left and right:
            if left[0] <= right[0]:
                sorted_problem.append(left.pop(0))
            else:
                sorted_problem.append(right.pop(0))
        while left:
            sorted_problem.append(left.pop(0))
        while right:
            sorted_problem.append(right.pop(0))
    problem = sorted_problem


def quick_sort(problem):
    def partition(low, high):
        pivot = problem[high]  # Choose the last element as the pivot
        i = low - 1  # Pointer for the smaller element
        for j in range(low, high):
            if problem[j] < pivot:
                i += 1
                problem[i], problem[j] = problem[j], problem[i]  # Swap elements
        problem[i + 1], problem[high] = problem[high], problem[i + 1]  # Place pivot in the correct position
        return i + 1

    def quick_sort_recursive(low, high):
        while low < high:
            pivot_index = partition(low, high)
            if pivot_index - low < high - pivot_index:
                quick_sort_recursive(low, pivot_index - 1)
                low = pivot_index + 1
            else:
                quick_sort_recursive(pivot_index + 1, high)
                high = pivot_index - 1

    quick_sort_recursive(0, len(problem) - 1)


def heap_sort(problem):
    def heapify(problem, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and problem[i] < problem[left]:
            largest = left
        if right < n and problem[largest] < problem[right]:
            largest = right
        if largest != i:
            (problem[i], problem[largest]) = (problem[largest], problem[i])  # swap
            heapify(problem, n, largest)
    n = len(problem)
    for i in range(n // 2, -1, -1):
        heapify(problem, n, i)
    for i in range(n - 1, 0, -1):
        (problem[i], problem[0]) = (problem[0], problem[i])  # swap
        heapify(problem, i, 0)


def counting_sort(problem):
    max_el = max(problem)
    counter = [0 for _ in range(max_el + 1)]
    # Count occurrences of each element
    for el in problem:
        counter[el] += 1
    # Compute the prefix sums
    for i in range(1, max_el + 1):
        counter[i] += counter[i - 1]
    # Initialize the solution list with the same length as the problem
    solution = [0] * len(problem)
    # Place the elements in their sorted positions (iterating from right to left for stability)
    for el in problem[::-1]:
        solution[counter[el] - 1] = el
        counter[el] -= 1
    # Copy the sorted solution back into the problem list
    for i in range(len(problem)):
        problem[i] = solution[i]

def radix_sort(problem):
    max_el = max(problem)  # Find the maximum element to know the number of digits
    digit = 1  # Start with the least significant digit
    while max_el // digit > 0:
        size = len(problem)
        solution = [0] * size  # Temporary array to store sorted numbers based on current digit
        count = [0] * 10  # There are 10 possible digits (0-9)
        # Count the occurrences of digits at the current place value
        for i in range(size):
            index = problem[i] // digit
            count[index % 10] += 1
        # Compute the cumulative count (prefix sum)
        for i in range(1, 10):
            count[i] += count[i - 1]
        # Build the solution array (sorted by current digit), moving right to left for stability
        i = size - 1
        while i >= 0:
            index = problem[i] // digit
            solution[count[index % 10] - 1] = problem[i]
            count[index % 10] -= 1
            i -= 1
        # Copy the solution array to the original list (problem)
        for i in range(size):
            problem[i] = solution[i]
        # Move to the next more significant digit
        digit *= 10


def shell_sort(problem):
    n = len(problem)
    gap = n // 2  # Use integer division for the gap
    # Continue reducing the gap and sorting the subarrays
    while gap > 0:
        # Perform insertion sort with the given gap
        for i in range(gap, n):
            ordered_val = problem[i]
            j = i
            while j >= gap and problem[j - gap] > ordered_val:
                problem[j] = problem[j - gap]
                j -= gap
            problem[j] = ordered_val
        gap //= 2  # Reduce the gap using integer division


def cocktail_shaker_sort(problem):
    for i in range(len(problem) - 1, 0, -1):
        is_swapped = False
        for j in range(i, 0, -1):
            if problem[j] < problem[j - 1]:
                problem[j], problem[j - 1] = problem[j - 1], problem[j]
                is_swapped = True
        for j in range(i):
            if problem[j] > problem[j + 1]:
                problem[j], problem[j + 1] = problem[j + 1], problem[j]
                is_swapped = True
        if not is_swapped:
            break


def gnome_sort(problem):
    index = 0
    while index < len(problem):
        if index == 0:
            index = index + 1
        if problem[index] >= problem[index - 1]:
            index = index + 1
        else:
            problem[index], problem[index - 1] = problem[index - 1], problem[index]
            index = index - 1


def pancake_sort(problem):
    arr_len = len(problem)
    while arr_len > 1:
        mi = problem.index(max(problem[0:arr_len]))
        problem = problem[mi::-1] + problem[mi + 1:len(problem)]
        problem = problem[arr_len - 1::-1] + problem[arr_len:len(problem)]
        arr_len -= 1


def bogo_sort(problem):
    def is_sorted(problem):
        if len(problem) < 2:
            return True
        for i in range(len(problem) - 1):
            if problem[i] > problem[i + 1]:
                return False
        return True
    while not is_sorted(problem):
        rd.shuffle(problem)


def comb_sort(problem):
    shrink_fact = 1.3
    gaps = len(problem)
    swapped = True
    i = 0
    while gaps > 1 or swapped:
        gaps = int(float(gaps) / shrink_fact)
        swapped = False
        i = 0
        while gaps + i < len(problem):
            if problem[i] > problem[i + gaps]:
                problem[i], problem[i + gaps] = problem[i + gaps], problem[i]
                swapped = True
            i += 1


def tim_sort(problem):
    def calc_min_run(n):
        r = 0
        while n >= 32:
            r |= n & 1
            n >>= 1
        return n + r
    def one_insertion(arr, left, right):
        for i in range(left + 1, right + 1):
            j = i
            while j > left and arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1
    def merge(arr, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = [], []
        for i in range(0, len1):
            left.append(arr[l + i])
        for i in range(0, len2):
            right.append(arr[m + 1 + i])
        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len1:
            arr[k] = left[i]
            k += 1
            i += 1
        while j < len2:
            arr[k] = right[j]
            k += 1
            j += 1
    n = len(problem)
    min_run = calc_min_run(n)
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        one_insertion(problem, start, end)
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            if mid < right:
                merge(problem, left, mid, right)
        size = 2 * size


def sleep_sort(problem):
    def sleep_and_print(n, max_value):
        time.sleep(n * 0.001)  # Scale sleep time down to 1% of the number value
        # print(n, end=' ') <- this would be the returned array, but that's not really pretty for the display
    if not problem:
        return
    max_value = max(problem)
    threads = []
    for num in problem:
        thread = threading.Thread(target=sleep_and_print, args=(num, max_value))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def stooge_sort(problem):
    def stooge(arr, i, h):
        if i >= h:
            return
        # If first element is larger than the last, swap them
        if arr[i] > arr[h]:
            arr[i], arr[h] = arr[h], arr[i]
        # If there are more than 2 elements in the subarray
        if h - i + 1 > 2:
            t = (h - i + 1) // 3  # Compute one-third of the current segment length
            # Recursively sort the first 2/3 of the array
            stooge(arr, i, h - t)
            # Recursively sort the last 2/3 of the array
            stooge(arr, i + t, h)
            # Recursively sort the first 2/3 of the array again
            stooge(arr, i, h - t)
    stooge(problem, 0, len(problem) - 1)  # Call stooge on the entire array


def bitonic_sort(problem):
    def comp_and_swap(a, i, j, dire):
        if (dire == 1 and a[i] > a[j]) or (dire == 0 and a[i] > a[j]):
            a[i], a[j] = a[j], a[i]
    def bitonic_merge(a, low, cnt, dire):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                comp_and_swap(a, i, i + k, dire)
            bitonic_merge(a, low, k, dire)
            bitonic_merge(a, low + k, k, dire)
    def sort(a, low, cnt, dire):
        if cnt > 1:
            k = cnt // 2
            sort(a, low, k, 1)
            sort(a, low + k, k, 0)
            bitonic_merge(a, low, cnt, dire)
    sort(problem, 0, len(problem), 1)


def odd_even_sort(problem):
    is_sorted = 0
    while is_sorted == 0:
        is_sorted = 1
        temp = 0
        for i in range(1, len(problem) - 1, 2):
            if problem[i] > problem[i + 1]:
                problem[i], problem[i + 1] = problem[i + 1], problem[i]
                is_sorted = 0

        for i in range(0, len(problem) - 1, 2):
            if problem[i] > problem[i + 1]:
                problem[i], problem[i + 1] = problem[i + 1], problem[i]
                is_sorted = 0


def pigeonhole_sort(problem):
    my_min = min(problem)
    my_max = max(problem)
    size = my_max - my_min + 1
    holes = [0] * size
    for x in problem:
        holes[x - my_min] += 1
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            problem[i] = count + my_min
            i += 1


def smooth_sort(problem):
    def leonardo(k):
        if k < 2:
            return 1
        return leonardo(k - 1) + leonardo(k - 2) + 1
    def heapify(start, end):
        i = start
        j = 0
        k = 0
        while k < end - start + 1:
            if k & 0xAAAAAAAA:
                j = j + i
                i = i >> 1
            else:
                i = i + j
                j = j >> 1
            k = k + 1
        while i > 0:
            j = j >> 1
            k = i + j
            while k < end:
                if problem[k] > problem[k - i]:
                    break
                problem[k], problem[k - i] = problem[k - i], problem[k]
                k = k + i
            i = j
    p = len(problem) - 1
    q = p
    r = 0
    # back into an array
    while p > 0:
        if (r & 0x03) == 0:
            heapify(r, q)
        if leonardo(r) == p:
            r = r + 1
        else:
            r = r - 1
            q = q - leonardo(r)
            heapify(r, q)
            q = r - 1
            r = r + 1
        problem[0], problem[p] = problem[p], problem[0]
        p = p - 1
    for i in range(len(problem) - 1):
        j = i + 1
        while j > 0 and problem[j] < problem[j - 1]:
            problem[j], problem[j - 1] = problem[j - 1], problem[j]
            j = j - 1



# Search Algorithms
def linear_search(problem, target):
    for i, el in enumerate(problem):
        if el == target:
            return i
        else:
            return -1


def binary_search(problem, target, left=0, right=None):
    if right is None:
        right = len(problem) - 1
    if left > right:
        return -1
    divider = (left + right) // 2
    if problem[divider] == target:
        return divider
    elif problem[divider] < target:
        return binary_search(problem, target, divider + 1, right)
    else:
        return binary_search(problem, target, left, divider - 1)


def jump_search(problem, target):
    n = len(problem)
    step = math.sqrt(len(problem))
    prev = 0
    while problem[int(min(step, n) - 1)] < target:
        prev = step
        step += math.sqrt(n)
        if prev >= n:
            return -1
    while problem[int(prev)] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    # If element is found
    if problem[int(prev)] == target:
        return prev
    return -1


def exponential_search(problem, target):
    def bin_search(problem, l, r, target):
        if r >= l:
            mid = l + (r - l) // 2
            if problem[mid] == target:
                return mid
            if problem[mid] > target:
                return bin_search(problem, l,
                                    mid - 1, target)
            return bin_search(problem, mid + 1, r, target)
        return -1
    if problem[0] == target:
        return 0
    i = 1
    n = len(problem)
    while i < n and problem[i] <= target:
        i = i * 2
    return bin_search(problem, i // 2, min(i, n - 1), target)


def interpolation_search(problem, target):
    def inter_search(problem, lo, hi, target):
        # Check if target is within bounds
        if lo <= hi and problem[lo] <= target <= problem[hi]:
            # Avoid division by zero
            if problem[hi] == problem[lo]:
                if problem[lo] == target:
                    return lo
                return -1
            # Calculate the position
            pos = lo + ((hi - lo) // (problem[hi] - problem[lo]) * (target - problem[lo]))          
            # Check if target is found
            if problem[pos] == target:
                return pos
            # If target is larger, search in the right subarray
            if problem[pos] < target:
                return inter_search(problem, pos + 1, hi, target)
            # If target is smaller, search in the left subarray
            return inter_search(problem, lo, pos - 1, target)      
        return -1  # Target is not found
    lo = 0
    hi = len(problem) - 1
    return inter_search(problem, lo, hi, target)  # Return the result



def fibonacci_search(problem, target):
    n = len(problem)
    fib_sec = 0
    fib_first = 1
    fib_prime = fib_sec + fib_first
    while (fib_prime < n):
        fib_sec = fib_first
        fib_first = fib_prime
        fib_prime = fib_sec + fib_first
    offset = -1
    while (fib_prime > 1):
        i = min(offset + fib_sec, n - 1)
        if (problem[i] < target):
            fib_prime = fib_first
            fib_first = fib_sec
            fib_sec = fib_prime - fib_first
            offset = i
        elif (problem[i] > target):
            fib_prime = fib_sec
            fib_first = fib_first - fib_sec
            fib_sec = fib_prime - fib_first
        else:
            return i
    if (fib_first and problem[n - 1] == target):
        return n - 1
    return -1


def ternary_search(problem, target):
    def search(l, r, ar, key):
        if r >= l:
            mid1 = l + (r - l) // 3
            mid2 = r - (r - l) // 3
            if ar[mid1] == key:
                return mid1
            if ar[mid2] == key:
                return mid2
            if key < ar[mid1]:
                return search(l, mid1 - 1, ar, key)
            elif key > ar[mid2]:
                return search(mid2 + 1, r, ar, key)
            else:
                return search(mid1 + 1, mid2 - 1, ar, key)
        return -1
    
    return search(0, len(problem) - 1, problem, target)

def binary_search_tree(problem, target):
    class TreeNode:
        def __init__(self, key, index):
            self.left = None
            self.right = None
            self.value = key
            self.index = index
    def insert(root, key, index):
        if root is None:
            return TreeNode(key, index)
        else:
            if root.value < key:
                root.right = insert(root.right, key, index)
            else:
                root.left = insert(root.left, key, index)
        return root
    def search_bst_node(root, target):
        if root is None:
            return -1
        if root.value == target:
            return root.index
        elif root.value < target:
            return search_bst_node(root.right, target)
        else:
            return search_bst_node(root.left, target)
    root = None
    for i, value in enumerate(problem):
        root = insert(root, value, i)
    return search_bst_node(root, target)


def bidirectional_search(problem, target):
    left, right = 0, len(problem) - 1
    while left <= right:
        if problem[left] == target:
            return left
        if problem[right] == target:
            return right
        left += 1
        right -= 1
    return -1


def jump_point_search(problem, target):
    length = len(problem)
    step = int(math.sqrt(length))
    prev = 0

    while problem[min(step, length) - 1] < target:
        prev = step
        step += int(math.sqrt(length))
        if prev >= length:
            return -1

    for i in range(prev, min(step, length)):
        if problem[i] == target:
            return i
    return -1


def genetic_algorithm_search(problem, target):
    population = [rd.randint(0, len(problem) - 1) for _ in range(10)]
    for _ in range(100):  # 100 generations
        population = sorted(population, key=lambda x: abs(problem[x] - target))
        if problem[population[0]] == target:
            return population[0]
        population = [rd.randint(0, len(problem) - 1) for _ in range(10)]
    return -1

def simulated_annealing(problem, target):
    current_index = rd.randint(0, len(problem) - 1)
    for temperature in range(100, 0, -1):
        if problem[current_index] == target:
            return current_index
        next_index = rd.randint(0, len(problem) - 1)
        delta_e = abs(problem[next_index] - target) - abs(problem[current_index] - target)
        if delta_e < 0 or math.exp(-delta_e / temperature) > rd.random():
            current_index = next_index
    return -1


