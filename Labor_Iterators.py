class CustomIterator:
    def __init__(self, sequence):
        self.sequence = sequence
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            element = self.sequence[self.index]
            self.index += 1
            return element
        except IndexError:
            raise StopIteration


if __name__ == "__main__":
    names = ("John", "Jane", "Bob", "Alice")
    print(names[2])
    name_list = list(names)  # There are some wanted changes to the name_list in the future
    iterator = iter(name_list)
    while True:
        try:
            print(next(iterator), end=" ")
        except StopIteration:
            break
    print()

    c_i = CustomIterator((1,2,3,4,5,6,7,8,9,10))
    for num in c_i:
        print(num, end= " ")
