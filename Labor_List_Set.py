food = ["pizza", "sushi", "tacos", "salat", "fruit salat", "sandwich", "chicken parmesan", "spaghetti", "french fries", "ice cream"]
fruits = [meal for meal in food if meal.lower().endswith("fruit")]
print(fruits)
fruit_set = set(fruits)  # its a set of fruits, it doesn't have to be ordered
if "fruit salat" in fruit_set:
    print("Yes, fruit salat also contains fruits.")

vegetables = {"tomato", "kale", "broccoli", "carrot"}
combi_set = vegetables.union({"potato"}).intersection({"broccoli", "carrot"}).difference({"carrot"})
print(combi_set)


