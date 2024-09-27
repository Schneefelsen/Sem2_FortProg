from random import shuffle
import util
import multiprocessing
import time





"""
This is a testing script for all used search and sort functions.
"""

if __name__ == "__main__":
    array = [i for i in range(5000)]
    shuffle(array)
    for key in util.name_to_function_dict().keys():
        if "Sort" in key:
            solve = getattr(util, util.name_to_function_dict()[key])
            p = multiprocessing.Process(target=solve, name="Solve", args=(array,))
            p.start()
            p.join(10)
            if p.is_alive():
                p.terminate()
                p.join()
                print(f"{key}: Timeout!")
            print(f"{key}: done")
        else:
            solve = getattr(util, util.name_to_function_dict()[key])
            p = multiprocessing.Process(target=solve, name="Solve", args=(array, 4998,))
            p.start()
            p.join(10)
            if p.is_alive():
                p.terminate()
                p.join()
                print(f"{key}: Timeout!")
            print(f"{key}: done")
