import numpy as np
import time
import random as rd
import util


class StatCard:
    """
    This class generates and safes the stats of a sort or search algorithm in one place and displays them.
    """
    def __init__(self, name: str, runtime: str, complexity: str, usage: str, fame: str):
        self.name = name
        self.runtime = int(runtime)
        self.complexity = int(complexity)
        self.usage = int(usage)
        self.fame = int(fame)

    def get_stats(self) -> [str, int, int, int]:
        return [self.name, self.runtime, self.complexity, self.usage]

    def print_card(self):
        space = 30
        print(f"+{'-' * (space)}+")
        print(f"|{self.name.center(space)}|")
        print(f"|{' ' * (space)}|")
        print(f"| Runtime: {('★' * self.runtime + '☆' * (10 - self.runtime)).rjust(space - 13)}|")
        print(f"| Complexity: {('★' * self.complexity + '☆' * (10 - self.complexity)).rjust(space - 16)}|")
        print(f"| Usage: {('★' * self.usage + '☆' * (10 - self.usage)).rjust(space - 11)}|")
        print(f"| Fame: {('★' * self.fame + '☆' * (10 - self.fame)).rjust(space - 10)}|")
        print(f"+{'-' * (space)}+")


class SortiGame:
    """
    This is a Game about choosing the right sorting algorithm for the right job
    and racing your friend or the computer with it.
    """
    def __init__(self):
        self.is_search_algorithm = 0  # 0: sort, 1: search
        self.card_list = []
        self.players = [0, 0]
        self.hand_size = 1
        self.player_hands = [[], []]
        self.scores = {0: 0, 1: 0}

    def get_cards(self, file_path: str):
        with open(file_path) as f:
            next(f)
            for line in f:
                stats = line.split(",")
                self.card_list.append(StatCard(*stats))

    def show_hand(self, player: int):
        for card in self.player_hands[player]:
            card.print_card()

    def menu(self):
        pass

    def choose_fighter(self, player: int) -> StatCard:
        while len(self.player_hands[player]) < self.hand_size:  # fill hand first
            self.player_hands[player].append(self.card_list.pop())
        if self.players[player]:  # human
            input(f"Player {player + 1}, please press Enter:\n> ")
            for i, card in enumerate(self.player_hands[player]):
                print(f"[{i+1}]")
                card.print_card()
            print("Choose your fighter:")
            while True:
                try:
                    chosen_card = int(input("> ")) - 1
                    if chosen_card >= self.hand_size:
                        raise ValueError
                    return self.player_hands[player].pop(chosen_card)
                except ValueError:
                    print("Use the card number of your choice.")
        else:  # easy com
            print(f"Player {player + 1} makes its choice.\n> ")
            time.sleep(1.5)
            chosen_card = rd.randint(0, self.hand_size)
            return self.player_hands[player].pop(chosen_card)

    def play_turn(self):
        fighting_cards = []
        length = 50000
        fighting_stage = [i for i in range(length)]

        rd.shuffle(self.card_list)
        fighting_cards.append(self.choose_fighter(0))
        fighting_cards.append(self.choose_fighter(1))
        print("Introducing the Fighters!")
        time.sleep(1)
        print(f"In Player Ones' corner: {fighting_cards[0].get_stats()[0]}")
        time.sleep(1)
        print(f"In Player Twos' corner: {fighting_cards[1].get_stats()[0]}")
        time.sleep(1)
        print("Let's pit both of them against each other!!!")
        if self.is_search_algorithm:
            rd.shuffle(fighting_stage)
            print("The stage is a unsorted List of length 50000! 10 Seconds max.")
        else:
            print("The stage is a sorted List of length 50000! 10 Seconds max.")
        print("3!")
        time.sleep(1)
        print("2!")
        time.sleep(1)
        print("1!")
        time.sleep(1)
        print("Fight!!!")
        # TODO: multithreading and progress bar
        # TODO: identify winner by first completed thread?
        solve = getattr(util, util.name_to_function_dict()[fighting_cards[0].get_stats()[0]])
        solve(fighting_stage)
        solve = getattr(util, util.name_to_function_dict()[fighting_cards[1].get_stats()[0]])
        solve(fighting_stage)

    def play(self):
        print("Welcome to Sorti!")
        print("Do you want to test your knowledge on sorting or searching algorithms?")
        print("Do you want to let them clash in an epic battle of efficiency and speed?!")
        print("Tell me then: do you want to race a friend (1) or my glorious self (2)?")
        try:
            if int(input("> ").strip()) == 1:
                self.players = [1, 1]
            else:
                self.players = [1, 0]
        except ValueError:
            self.players = [1, 0]
        print("And do you want to search (1) or sort (2)?")
        try:
            if int(input("> ").strip()) == 1:
                self.is_search_algorithm = 1
        except ValueError:
            pass  # just stays 0
        algorithms = ["Sorting_Algorithms_with_stats.csv", "Search_Algorithms_with_stats.csv"][self.is_search_algorithm]
        self.get_cards(algorithms)
        self.play_turn()


if __name__ == "__main__":
    game = SortiGame()
    game.play()
