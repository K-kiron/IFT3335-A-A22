## Solve Every Sudoku Puzzle

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}
import itertools
import random
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))
             for s in squares)
boxs=[units['B2'][2],units['B5'][2],units['B8'][2],units['E2'][2],units['E5'][2],units['E8'][2],units['H2'][2],
          units['H5'][2],units['H8'][2]]
# print(boxs)
################ Unit Tests ################

def test():
    "A set of tests that must pass."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print ('All tests pass.')

################ Parse a Grid ################

def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s,d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False ## (Fail if we can't assign d to square s.)
    return values

def parse_grid_heuristique(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s,d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False ## (Fail if we can't assign d to square s.)
    
    #naked & hidden
    for u in unitlist:
        naked(values, u)
        hidden(values, u)
        
    return values

def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))

################ Constraint Propagation ################

def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False

def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values ## Already eliminated
    values[s] = values[s].replace(d,'')
    ## Single
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## Hidden Singles
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

################ Display as 2-D grid ################

def display(values):
    "Display these values as a 2-D grid."
    width = 1+max(len(values[s]) for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        linegrid=''.join(values[r+c].center(width)+('|' if c in '36' else '')for c in cols)
        print(linegrid)
        if r in 'CF':
            print(line)


################ Naked pair/triple ################
def naked(values, cur_units):
    pair_count = {}
    pair_map = {}
    triple_count = {}
    triple_map = {}

    for unit in cur_units:
        val = values[unit]
        if len(val) != 2:
            if len(val) == 3:
                if val in triple_count:
                    triple_count[val] += 1
                    triple_map[val].append(unit)
                else:
                    triple_count[val] = 1
                    triple_map[val] = [unit]

            continue
        if val in pair_count:
            pair_count[val] += 1
            pair_map[val].append(unit)
        else:
            pair_count[val] = 1
            pair_map[val] = [unit]


    for pair, count in pair_count.items():
        if count != 2:
            for triple, countTriple in triple_count.items():

                if all(p in triple for p in pair):
                    triple_count[triple] += count
                    triple_map[triple]=triple_map[triple]+pair_map[pair]
            continue
        for unit in cur_units:
            if unit in pair_map[pair]:
                continue
            values = eliminate(values, unit, pair[0])
            values = eliminate(values, unit, pair[1])

    for triple, countTriple in triple_count.items():
        if countTriple != 3:
            continue
        for unit in cur_units:
            if unit in triple_map[triple]:
                continue

            values = eliminate(values, unit, triple[0])
            values = eliminate(values, unit, triple[1])
            values = eliminate(values, unit, triple[2])


################ Hidden pair/triple ################

def hidden(values, cur_units):
    pair_count = {}
    pair_map = {}
    triple_count = {}
    triple_map = {}

    for unit in cur_units:
        val = values[unit]
        if len(val) != 2:
            if len(val) == 3:
                if val in triple_count:
                    triple_count[val] += 1
                    triple_map[val].append(unit)
                else:
                    triple_count[val] = 1
                    triple_map[val] = [unit]

            continue
        if val in pair_count:
            pair_count[val] += 1
            pair_map[val].append(unit)
        else:
            pair_count[val] = 1
            pair_map[val] = [unit]

    # for i in digits:
    #     hidden=False
    #     hiddenPositon = ''
    #     for unit in cur_units:
    #         if i in values[unit]:
    #             if hidden==False:
    #                 hidden=True
    #                 hiddenPositon = unit
    #             else:
    #                 hidden = False
    #                 break
    #     if hidden==True:
    #         values=assign(values,hiddenPositon,i)


################ Search ################
def calcul_conflit(values):
    conflit = 0
    for uni in unitlist[:18]:
        for i in range(9):
            for j in range(i + 1, 9):
                if values[uni[i]] == values[uni[j]]:
                    conflit += 1
    return conflit

def solve(grid): return search(parse_grid(grid))
def solveProfondeur(grid): return purProfondeurSearch(parse_grid(grid))
def solveHeuristique(grid): return search(parse_grid_heuristique(grid))

def solveHillClimbing(grid):
    values=parse_grid(grid)
    swapPairs=[]
    for box in boxs:
        elemNonFixe = []
        digitrest='123456789'
        for b in box:
            if len(values[b]) == 1:
                digitrest=digitrest.replace(values[b],'')


        for b in box:
            if len(values[b]) > 1:
                values[b]=random.choice(digitrest)
                digitrest=digitrest.replace(values[b],'')
                elemNonFixe.append(b)
        swapPairs = swapPairs+list(itertools.combinations(elemNonFixe,2))

        conflit=calcul_conflit(values)


    return search_hill_climbing(values,swapPairs,conflit)
    #return values




def search(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in squares):
        return values ## Solved!
    ## Chose the unfilled square s with the fewest possibilities
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))
                for d in values[s])

def purProfondeurSearch(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in squares):
        return values ## Solved!
    ## Chose the unfilled square s with the fewest possibilities
    for s in squares:
        if len(values[s]) > 1:
            a=s
            break
    return some(purProfondeurSearch(assign(values.copy(), a, d))
                for d in values[a])


def search_hill_climbing(values,swapPairs,conflit):
    "Using depth-first search and propagation, try all possible values."
    numConflit = conflit

    #print('numConflit: '+str(numConflit))
    if numConflit==0:
        #print('reussi')
        return values## Solved!
    ## Chose the unfilled square s with the fewest possibilities

    conflitList=[]
    for sp in swapPairs:
        conflitChange = 0
        for i in [0,1]:
            for unit in units[sp[i]][:2]:
                for u in unit:
                    if u != sp[i] and values[u] == values[sp[i]]:
                        conflitChange -= 1
                    j = -1 * i + 1
                    if u != sp[j] and values[u] == values[sp[j]]:
                        conflitChange += 1

        conflitList.append(conflitChange)

    min_index = conflitList.index(min(conflitList))

    if conflitList[min_index] == 0:
        #print('Arrete, ne trouve pas de solution')
        return False
    p=swapPairs[min_index]
    b = values[p[0]]
    values[p[0]] = values[p[1]]
    values[p[1]] = b
    #display(values)

    num_conflit = calcul_conflit(values)
    #print(swapPairs[min_index])
    #print("conflit changer: "+str(conflitList[min_index]))
    return search_hill_climbing(values,swapPairs,num_conflit)





################ Utilities ################

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False

def from_file(filename, sep='\n'):
    "Parse a file into a list of strings, separated by sep."
    return open(filename).read().strip().split(sep)

def shuffled(seq):
    "Return a randomly shuffled copy of the input sequence."
    seq = list(seq)
    random.shuffle(seq)
    return seq

################ System test ################

import time, random

def solve_all(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""
    def time_solve(grid):
        start = time.perf_counter()
        values = solve(grid)
        t = time.perf_counter()-start
        ## Display puzzles that take long enough
        if showif is not None and t > showif:
            display(grid_values(grid))
            if values: display(values)
            print ('(%.2f seconds)\n' % t)
        return (t, solved(values))
    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print ("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs)." % (
            sum(results), N, name, sum(times)/N, N/sum(times), max(times)))


def solve_all_pur_profondeur(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""
    def time_solve(grid):
        start = time.perf_counter()
        values = solveProfondeur(grid)
        t = time.perf_counter()-start
        ## Display puzzles that take long enough
        if showif is not None and t > showif:
            display(grid_values(grid))
            if values: display(values)
            print ('(%.2f seconds)\n' % t)
        return (t, solved(values))
    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print ("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs by Deepth First Search)." % (
            sum(results), N, name, sum(times)/N, N/sum(times), max(times)))

def solve_all_heuristique(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""
    def time_solve(grid):
        start = time.perf_counter()
        values = solveHeuristique(grid)
        t = time.perf_counter()-start
        ## Display puzzles that take long enough
        if showif is not None and t > showif:
            display(grid_values(grid))
            if values: display(values)
            print ('(%.2f seconds)\n' % t)
        return (t, solved(values))
    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print ("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs by Heuristics Search)." % (
            sum(results), N, name, sum(times)/N, N/sum(times), max(times)))

def solve_all_hill_climbing(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""
    def time_solve(grid):
        start = time.perf_counter()
        values = solveHillClimbing(grid)
        t = time.perf_counter()-start
        ## Display puzzles that take long enough
        if showif is not None and t > showif:
            display(grid_values(grid))
            if values: display(values)
            print ('(%.2f seconds)\n' % t)
        return (t, solved(values))
    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print ("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs by hill climbing)." % (
            sum(results), N, name, sum(times)/N, N/sum(times), max(times)))


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."
    def unitsolved(unit): return set(values[s] for s in unit) == set(digits)
    return values is not False and all(unitsolved(unit) for unit in unitlist)

def random_puzzle(N=17):
    """Make a random puzzle with N or more assignments. Restart on contradictions.
    Note the resulting puzzle is not guaranteed to be solvable, but empirically
    about 99.8% of them are solvable. Some have multiple solutions."""
    values = dict((s, digits) for s in squares)
    for s in shuffled(squares):
        if not assign(values, s, random.choice(values[s])):
            break
        ds = [values[s] for s in squares if len(values[s]) == 1]
        if len(ds) >= N and len(set(ds)) >= 8:
            return ''.join(values[s] if len(values[s])==1 else '.' for s in squares)
    return random_puzzle(N) ## Give up and make a new puzzle

grid1  = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
grid2  = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
hard1  = '.....6....59.....82....8....45........3........6..3.54...325..6..................'
#display(parse_grid(grid2))
#solveHillClimbing(grid2)
#display(solve(grid2))
#display(solveHillClimbing(grid2))
if __name__ == '__main__':
    test()
    
    # solve_all(from_file("top95.txt"), "top95", None)
    solve_all(from_file("100sudoku.txt"),"easy", None)
    # solve_all(from_file("1000sudoku.txt"),"hard", None)
    # solve_all([random_puzzle() for _ in range(99)], "random", 100.0)
    # solve_all([random_puzzle() for _ in range(999)], "random", 1000.0)

    # print("Solving by Deepth First Search")
    # solve_all_pur_profondeur(from_file("top95.txt"), "top95", None)
    solve_all_pur_profondeur(from_file("100sudoku.txt"),"easy", None)
    # solve_all_pur_profondeur(from_file("1000sudoku.txt"),"hard", None)
    # solve_all_pur_profondeur([random_puzzle() for _ in range(99)], "random", 100.0)
    # solve_all_pur_profondeur([random_puzzle() for _ in range(999)], "random", 1000.0)

    # print("Solving by Heuristics Search")
    # solve_all_heuristique(from_file("top95.txt"), "top95", None)
    solve_all_heuristique(from_file("100sudoku.txt"),"easy", None)
    # solve_all_heuristique(from_file("1000sudoku.txt"),"hard", None)
    # solve_all_heuristique([random_puzzle() for _ in range(99)], "random", 100.0)
    # solve_all_heuristique([random_puzzle() for _ in range(999)], "random", 1000.0)

    # print("Solving by Hill Climbing")
    # solve_all_hill_climbing(from_file("top95.txt"), "top95", None)
    solve_all_hill_climbing(from_file("100sudoku.txt"),"easy", None)
    # solve_all_hill_climbing(from_file("1000sudoku.txt"),"hard", None)
    # solve_all_hill_climbing([random_puzzle() for _ in range(99)], "random", 100.0)
    # solve_all_hill_climbing([random_puzzle() for _ in range(999)], "random", 1000.0)
    

    #solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("top95.txt"), "hard", None)
    # solve_all(from_file("hardest.txt"), "hardest", None)
    # solve_all([random_puzzle() for _ in range(99)], "random", 100.0)
    #solve_all(from_file("1000sudoku.txt"),"easy", None)
    #solve_all_pur_profondeur(from_file("1000sudoku.txt"), "easy", None)


## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/
