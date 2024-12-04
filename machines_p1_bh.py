import numpy as np
import random
from itertools import product

import time
import math
import copy

class MCTSNode:
    def __init__(self, board, available_pieces, parent=None, move=None):
        self.board = board
        self.available_pieces = available_pieces
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.move_count = 0


class P2:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.simulation_count = 100  # Default : 1,000
        self.exploration_constant = 0.75

    def select_piece(self):
        root = MCTSNode(self.board, self.available_pieces)

        # for _ in range(self.simulation_count):
        #     node = self.select(root, None)
        #     value = 1 - self.simulate(node)
        #     self.backpropagate(node, value)

        # 상대에게 불리한 말을 계산
        opponent_unfavorable_piece = self.find_opponent_unfavorable_piece(root)

        return opponent_unfavorable_piece

    def find_opponent_unfavorable_piece(self, node):
        """
        상대에게 가장 불리한 말을 선택하는 함수.
        """
        max_disruption = float('-inf')
        best_piece = None

        for piece in self.available_pieces:
            disruption_score = self.calculate_disruption_score(node.board, piece)
            if disruption_score > max_disruption:
                max_disruption = disruption_score
                best_piece = piece
        print("--------------------------")
        return best_piece
    
    def calculate_disruption_score(self, board, piece):
        """
        주어진 말이 상대에게 얼마나 방해가 되는지 점수를 계산.
        """
        score = 0

        # 1. 상대가 유리한 줄(행, 열, 대각선)을 방해하는가?
        for line in self.get_all_lines(board):
            if 0 in line:  # 줄에 빈 칸이 있는 경우만 평가
                potential_line = [self.pieces.index(piece) + 1 if cell == 0 else cell for cell in line]
                if self.check_line(potential_line):  # 상대가 이 줄을 완성할 수 있는가?
                    score -= 10  # 상대의 완성을 방해할 수 있다면 높은 점수를 부여

        # 2. 남은 말을 줄이는 효과
        if piece in self.available_pieces:
            score += 5  # 상대가 선택할 말을 줄임

        print(score)
        print('----')
        return score
    
    def get_all_lines(self, board):
        """
        보드의 모든 줄(행, 열, 대각선, 2x2 서브그리드)을 반환.
        """
        lines = []

        # 행, 열 추가
        for r in range(4):
            lines.append([board[r][c] for c in range(4)])
        for c in range(4):
            lines.append([board[r][c] for r in range(4)])

        # 대각선 추가
        lines.append([board[i][i] for i in range(4)])
        lines.append([board[i][3 - i] for i in range(4)])

        # 2x2 서브그리드 추가
        for r in range(3):
            for c in range(3):
                lines.append([board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]])

        return lines




    def place_piece(self, selected_piece):
        root = MCTSNode(self.board, self.available_pieces)
        #print(selected_piece)  # test_page
        
        for _ in range(self.simulation_count):
            #move_count = 0  # Manage all process count
            node = self.select(root, selected_piece)
            value = self.simulate(node)
            self.backpropagate(node, value)

        best_child = max(root.children, key=lambda c: c.visits)
        # print("root's value: " + str(root.value))  # test_page


        return best_child.move[1]

    def select(self, node, piece):
        if piece is None:
            piece = random.choice(self.available_pieces) # 줄 말을 랜덤하게 뽑을지?    

        while node.children:
            # print("children exist")  # test_pages
            '''
            # Additional child node extensions
            if not all(child.visits > 0 for child in node.children):
                print("[Message] children exist but not visits")  # test_pages
                return self.expand(node, piece)
            '''
            node = self.uct_select(node)
            #return node  # test_page
            # break  # test_page

        #print("[Message] no children")  # test_page
        return self.expand(node, piece)

    def expand(self, node, piece):
        if self.is_terminal(node.board):
            return node

        for row, col in product(range(4), repeat=2):
            if node.board[row][col] == 0:
                new_board = copy.deepcopy(node.board)

                new_board[row][col] = self.pieces.index(piece) + 1
                new_available_pieces = node.available_pieces[:]

                #print(f"add children in row: {row} col: {col}")  # test_page
                child = MCTSNode(new_board, new_available_pieces, node, (piece, (row, col)))
                node.children.append(child)

        return self.uct_select(node)
        # return random.choice(node.children)

    def simulate(self, node):
        board = copy.deepcopy(node.board)
        available_pieces = node.available_pieces[:]
        move_count = 0

        while not self.is_terminal(board) and available_pieces:
            piece = random.choice(available_pieces)
            empty_cells = [(r, c) for r, c in product(range(4), repeat=2) if board[r][c] == 0]

            if not empty_cells:
                break

            row, col = random.choice(empty_cells)
            board[row][col] = self.pieces.index(piece) + 1
            available_pieces.remove(piece)
            move_count = move_count + 1

        return self.evaluate(move_count)

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            #print(f"node's visit: {node.visits:>4}")  # test_page
            node.value += value
            #print(f"node's value: {node.value:>4}")  # test_page
            node = node.parent
        #print("--")

    def uct_select(self, node):
        for child in node.children:
            if child.visits == 0:
                return child

        return max(node.children, key=lambda c: c.value / c.visits + 2 * self.exploration_constant * math.sqrt(
                                                math.log(node.visits) / c.visits))

    def is_terminal(self, board):
        return self.check_win(board) or all(board[r][c] != 0 for r, c in product(range(4), repeat=2))

    def check_line(self, line):
        if 0 in line:
            return False  # Incomplete line

        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])

        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True

        return False

    def check_2x2_subgrid_win(self, board):
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]

                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]

                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True

        return False

    def check_win(self, board):
        # Check rows, columns, and diagonals
        for col in range(4):
            if self.check_line([board[row][col] for row in range(4)]):
                return True

        for row in range(4):
            if self.check_line([board[row][col] for col in range(4)]):
                return True

        if self.check_line([board[i][i] for i in range(4)]) or self.check_line([board[i][3 - i] for i in range(4)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win(board):
            return True

        return False

    def evaluate(self, count):
        if count % 2 == 0:
            return 1  # 승
        else:
            return 0  # 패