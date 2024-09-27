import os
import time
import multiprocessing
import random as rd
import util


class StatCard:
    """
    This class generates and stores the stats of a sorting or searching algorithm.
    It also provides methods to display these stats in a formatted manner.
    
    Attributes:
        name (str): The name of the algorithm.
        runtime (int): The runtime of the algorithm (represented as a score).
        complexity (int): The complexity of the algorithm (represented as a score).
        usage (int): The usage frequency of the algorithm (represented as a score).
        fame (int): The fame of the algorithm (represented as a score).
    """

    def __init__(self, name: str, runtime: str, complexity: str, usage: str, fame: str):
        self.name = name
        self.runtime = int(runtime)
        self.complexity = int(complexity)
        self.usage = int(usage)
        self.fame = int(fame)

    def get_stats(self) -> [str, int, int, int]:
        """Returns the stats of the algorithm as a list."""
        return [self.name, self.runtime, self.complexity, self.usage]

    def print_card(self):
        """Displays the stat card in a formatted way."""
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
    This class implements a game about choosing the fastest sorting or searching algorithm and 
    racing against another player or the computer.

    Attributes:
        is_search_algorithm (int): Indicates whether the game is about searching (1) or sorting (0).
        card_list (list): A list of StatCard objects representing available algorithms.
        players (list): A list representing two players (0: computer, 1: human).
        hand_size (int): Number of cards each player can hold in hand.
        player_hands (list): List containing players' hands (lists of StatCard).
        scores (dict): Dictionary to hold scores for both players.
    """

    def __init__(self):
        self.is_search_algorithm = 0  # 0: sort, 1: search
        self.card_list = []
        self.players = [0, 0]
        self.hand_size = 3
        self.player_hands = [[], []]
        self.scores = {0: 0, 1: 0}

    def get_cards(self, file_path: str):
        """
        Reads algorithm stats from the corresponding CSV file and populates the card_list with StatCard objects.
        
        Args:
            file_path (str): Path to the CSV file containing algorithm stats.
        """
        with open(file_path) as f:
            next(f)
            for line in f:
                stats = line.split(",")
                self.card_list.append(StatCard(*stats))

    def choose_fighter(self, round, player: int) -> StatCard:
        """
        Allows the player to choose a StatCard (algorithm) for the current round.
        
        Args:
            round (int): The current round number.
            player (int): The player number (0 or 1).
        
        Returns:
            StatCard: The chosen StatCard by the player.
        """
        if self.players[player]:  # human
            input(f"Player {player + 1}, please press Enter:\n> ")
            for i, card in enumerate(self.player_hands[player]):
                print(f"[{i+1}]")
                card.print_card()
            print("Choose your fighter:")
            while True:
                try:
                    chosen_card = int(input("> ")) - 1
                    if chosen_card not in range(len(self.player_hands[player])):
                        raise ValueError
                    return self.player_hands[player].pop(chosen_card)
                except ValueError:
                    print("Use the card number of your choice.")
        else:  # easy com
            print(f"Player {player + 1} makes its choice.\n> ")
            time.sleep(1.5)
            chosen_card = rd.randint(0, self.hand_size-1-round)
            return self.player_hands[player].pop(chosen_card)

    def play_turns(self):
        """
        Conducts three rounds of the game, where each round consists of selecting fighters 
        and comparing their performances using the chosen algorithms.
        """
        score = [0, 0]
        for round in range(3):
            fighting_cards = []
            length = 50000
            fighting_list = [i for i in range(length)]  # Create a list of 50000 elements
            fighting_cards.append(self.choose_fighter(round, 0))  # Player 1 chooses
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the screen for visual clarity
            fighting_cards.append(self.choose_fighter(round, 1))  # Player 2 chooses
            
            # Announce the fighters
            print(f"Round {round + 1}:")
            print("Introducing the Fighters!")
            time.sleep(1)
            print(f"In Player Ones' corner: {fighting_cards[0].get_stats()[0]}")
            time.sleep(1)
            print(f"In Player Twos' corner: {fighting_cards[1].get_stats()[0]}")
            time.sleep(1)
            print("Let's pit both of them against each other!!!")

            if self.is_search_algorithm:
                rd.shuffle(fighting_list)
                fighting_list_copy = fighting_list.copy()  # Ensure both algorithms work on the same initial list
                print("The stage is a unsorted List of length 50000! 10 Seconds max.")
            else:
                fighting_list_copy = fighting_list.copy()  # Ensure both algorithms work on the same initial list
                print("The stage is a sorted List of length 50000! 10 Seconds max.")
            
            time.sleep(1)
            print("3!")
            time.sleep(1)
            print("2!")
            time.sleep(1)
            print("1!")
            time.sleep(1)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Fight!!!")

            if self.is_search_algorithm:
                # compute first algorithm
                algorithm_1 = getattr(util, util.name_to_function_dict()[fighting_cards[0].get_stats()[0]])
                p = multiprocessing.Process(target=algorithm_1, name="A1", args=(fighting_list, 24999,))
                p.start()
                timer_1_start = time.perf_counter()
                p.join(10)  # force a timeout after 10 seconds
                if p.is_alive():
                    p.terminate()
                    p.join()
                timer_1_stop = time.perf_counter()
                timer_1 = timer_1_stop - timer_1_start
                # compute second algorithm
                algorithm_2 = getattr(util, util.name_to_function_dict()[fighting_cards[1].get_stats()[0]])
                p = multiprocessing.Process(target=algorithm_2, name="A2", args=(fighting_list_copy, 24999,))
                p.start()
                timer_2_start = time.perf_counter()
                p.join(10)  # force a timeout after 10 seconds
                if p.is_alive():
                    p.terminate()
                    p.join()
                timer_2_stop = time.perf_counter()
                timer_2 = timer_2_stop - timer_2_start
            else:
                # compute first algorithm
                algorithm_1 = getattr(util, util.name_to_function_dict()[fighting_cards[0].get_stats()[0]])
                p = multiprocessing.Process(target=algorithm_1, name="A1", args=(fighting_list,))
                p.start()
                timer_1_start = time.perf_counter()
                p.join(10)  # force a timeout after 10 seconds
                if p.is_alive():
                    p.terminate()
                    p.join()
                timer_1_stop = time.perf_counter()
                timer_1 = timer_1_stop - timer_1_start
                # compute second algorithm
                algorithm_2 = getattr(util, util.name_to_function_dict()[fighting_cards[1].get_stats()[0]])
                p = multiprocessing.Process(target=algorithm_2, name="A2", args=(fighting_list_copy,))
                p.start()
                timer_2_start = time.perf_counter()
                p.join(10)  # force a timeout after 10 seconds
                if p.is_alive():
                    p.terminate()
                    p.join()
                timer_2_stop = time.perf_counter()
                timer_2 = timer_2_stop - timer_2_start

            print(f"{fighting_cards[0].get_stats()[0]}: {timer_1:.10f}s")
            print(f"{fighting_cards[1].get_stats()[0]}: {timer_2:.10f}s")
            if self.players == [1, 0]:  # Player vs. Computer
                if timer_1 < timer_2:
                    print("This round is yours!")
                    score[0] += 1
                elif timer_1 == timer_2:
                    print("A tie, how did this even happen?!")
                else:
                    print("I won this one!")
                    score[1] += 1
            else:  # Player vs. Player
                if timer_1 < timer_2:
                    print("Player 1 won this one!")
                    score[0] += 1
                elif timer_1 == timer_2:
                    print("A tie, how did this even happen?!")
                else:
                    print("Player 2 won this one!")
                    score[1] += 1

        # Announce overall winner
        if self.players == [1, 0]:  # Player vs. Computer
            if score[0] < score[1]:
                print("HAHA, I won! But you may certainly try again if you like the feeling of losing.")
            elif score[0] == score[1]:
                print("I don't know how, but we tied... This never happened before...")
            else:
                print("Congratulations, you have bested me in a fair fight!")
        else:
            if score[0] < score[1]:  # Player vs. Player
                print("Player 2 won the competition! Congratulations!")
            elif score[0] == score[1]:
                print("I don't know how, but you tied... This never happened before...")
            else:
                print("Player 1 won the competition! Congratulations!")


    def play(self):
        """
        Main method to start the game. It prompts the user for their preferences, 
        initializes the game, and manages the game loop.
        """

        print("Welcome to Sorti!")
        print("Do you want to test your knowledge on sorting or searching algorithms?")
        print("Do you want to let them clash in an epic battle of efficiency and speed?!")
        while True:
            print("Tell me then: do you want to race a friend (1) or my glorious self (2)?")
            try:
                if int(input("> ").strip()) == 1:
                    self.players = [1, 1]  # Both players are human
                else:
                    self.players = [1, 0]  # One player is a computer
            except ValueError:
                self.players = [1, 0]  # Default to player vs. computer

            print("And do you want to search (1) or sort (2)?")
            try:
                if int(input("> ").strip().lower()) == 1:
                    self.is_search_algorithm = 1  # Set the game to search algorithms
            except ValueError:
                self.is_search_algorithm = 0  # Default to sorting algorithms
            
            # Choose the corresponding CSV file for the chosen game mode
            algorithms = ["Sorting_Algorithms_with_stats.csv", "Search_Algorithms_with_stats.csv"][self.is_search_algorithm]
            self.get_cards(algorithms)
            print("You'll play three rounds, so play your cards wisely...")
            rd.shuffle(self.card_list)
            
            # Deal cards to players
            for player in [0, 1]:
                while len(self.player_hands[player]) < self.hand_size:  # fill hand first
                    self.player_hands[player].append(self.card_list.pop())
            self.play_turns()
            print("Do you want to play again? (Y/N)")
            if input("> ").strip().lower() != "y":
                print("Have a nice day!")
                break

if __name__ == "__main__":
    game = SortiGame()
    game.play()
