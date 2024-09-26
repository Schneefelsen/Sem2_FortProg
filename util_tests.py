from random import shuffle
import util

if __name__ == "__main__":
    array = [i for i in range(5000)]
    shuffle(array)
    for key in util.name_to_function_dict().keys():
        if "Sort" in key:
            solve = getattr(util, util.name_to_function_dict()[key])
            print(f"{key}: {solve(array)}")
        else:
            solve = getattr(util, util.name_to_function_dict()[key])
            print(f"{key}: {solve(array, 4998)}")