from random import randint, random, choices, sample

# genetic algorithm
class GeneticAlgorithm:
    def __init__(self, chromosome_length, n_iter, population_size, cross_rate, mutate_rate):
        self.chromosome_length = chromosome_length
        self.n_iter = n_iter
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

    def run(self):
        population = [self.generate_chromosome() for _ in range(self.population_size)]
        best, best_eval = population[0], self.fitness(population[0])

        for gen in range(self.n_iter):
            scores = [self.fitness(c) for c in population]
            for i in range(self.population_size):
                if scores[i] > best_eval:
                    best, best_eval = population[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (gen, population[i], scores[i]))
            selected = self.selection(population, scores)
            children = list()

            # Apply crossover and mutation based on the crossover and mutation rates
            while len(children) < self.population_size:
                # Select random indexes for crossover
                idx1, idx2 = randint(0, self.population_size - 1), randint(0, self.population_size - 1)
                p1, p2 = selected[idx1], selected[idx2]
            
                # Apply crossover with a probability of cross_rate
                if random() < self.cross_rate:
                    for c in self.crossover(p1, p2):
                        # Apply mutation with a probability of mutate_rate
                        if random() < self.mutate_rate:
                            self.mutate(c)
                        children.append(c)
                else:
                    # If no crossover, check if we want to mutate
                    # then mutate and add or directly add parents to the next generation
                    if random() < self.mutate_rate:
                        self.mutate(p1)
                    if random() < self.mutate_rate:
                        self.mutate(p2)
                    children.extend([p1, p2])
                
                # Ensure children list does not exceed population
                if len(children) > self.population_size:
                    children = children[:self.population_size]

            population = children
        return best, best_eval


    """The objective function to evaluate fitness"""
    def fitness(self, chromosome):
        return sum(chromosome)

    """A function to generate a new chromosome."""
    def generate_chromosome(self):
        """Generates a new chromosome. This method can be overridden in subclasses"""
        raise NotImplementedError("Subclasses should implement this!")

    def selection(self, population, scores):
        raise NotImplementedError("Subclasses should implement this!")

    """ crossover: A function to perform crossover. """
    def crossover(self, p1, p2):
       raise NotImplementedError("Subclasses should implement this!")

    """mutation: A function to perform mutation."""
    def mutate(self, chromosome):
       raise NotImplementedError("Subclasses should implement this!")


""" 
Rank-Based Selection:
Sorts the population based on their fitness scores.
Assigns higher selection probabilities to individuals with better ranks.
This ensures a balanced selection favoring fitter individuals while keeping diversity.
"""      
class RankBasedSelection(GeneticAlgorithm):
    def __init__(self, chromosome_length, n_iter, population_size, cross_rate, mutate_rate):
        super().__init__(chromosome_length, n_iter, population_size, cross_rate, mutate_rate)

    def selection(self, pop, scores):
        # Sort the population based on fitness scores
        sorted_pop = [x for _, x in sorted(zip(scores, pop))]
        # Rank weights: higher rank has higher weight
        rank_weights = [i + 1 for i in range(self.population_size)]
        total_rank = sum(rank_weights)  # Total of rank weights
        # Calculate selection probabilities for each individual based on rank
        prob_select = [rank_weight / total_rank for rank_weight in rank_weights]
        # Select and return individuals based on calculated probabilities
        return choices(sorted_pop, weights=prob_select, k=self.population_size)


"""
Tournament Selection:
Randomly selects a subset (tournament) of the population.
Chooses the best individual from the tournament based on fitness scores.
This can vary the pressure of selection based on tournament size (parameter =k=).
"""
class TournamentSelection(GeneticAlgorithm):
    def __init__(self, chromosome_length, n_iter, population_size, cross_rate, mutate_rate, k=3):
        super().__init__( chromosome_length, n_iter, population_size, cross_rate, mutate_rate)
        self.k = k

    """Tournament selection"""    
    def selection(self, pop, scores):
        selected = []
        for _ in range(self.population_size):
            # Randomly select 'k' individuals for the tournament
            candidates = sample(range(self.population_size), self.k)
            # Find the best candidate with the minimal score in the tournament
            best_candidate_index = min(candidates, key=lambda index: scores[index])
            selected.append(pop[best_candidate_index])
        return selected



"""
Roulette Wheel Selection:
Calculates selection probabilities based on fitness scores directly.
Individuals with higher fitness scores have a greater chance of being selected.
Ensures all individuals have a chance of selection proportional to their fitness. 
"""
class RouletteWheelSelection(GeneticAlgorithm):
    def __init__(self, chromosome_length, n_iter, population_size, cross_rate, mutate_rate):
        super().__init__( chromosome_length, n_iter, population_size, cross_rate, mutate_rate)

    def selection(self, pop, scores):
        min_score = min(scores)
        if min_score < 0:
            # Adjust scores to be non-negative
            adjusted_scores = [score - min_score for score in scores]
        else:
            adjusted_scores = scores
        total_score = sum(adjusted_scores)
        # Calculate probability of selection for each individual
        prob_select = [score / total_score for score in adjusted_scores]
        # Select individuals based on calculated probabilities
        return choices(pop, weights=prob_select, k=self.population_size)



class MachineServicingProblem(RouletteWheelSelection):

    def __init__(self, chromosome_length, n_iter, population_size, cross_rate, mutate_rate, machines,widgets_per_machine,required_services_per_machine):
        super().__init__(chromosome_length, n_iter, population_size, cross_rate, mutate_rate)
        self.machines = machines
        self.widgets_per_machine = widgets_per_machine
        self.required_services_per_machine = required_services_per_machine
        self.chromosome_length = chromosome_length


    # Generate a random permutation of the cities
    def generate_chromosome(self):
        #return sample(self.cities, len(self.cities))
        #e.g [1,0,0,0,0,1,1,1]-period 1= [1,0,0,0], period 2 = [0,1,1,1]
        return [randint(0,1) for _ in range(self.chromosome_length)]

    def crossover(self, p1, p2):
        # Choose a random crossover point
        cross_idx = randint(1, self.chromosome_length - 1) 

        # Create offspring
        offspring1 = p1[:cross_idx] + p2[cross_idx:]
        offspring2 = p2[:cross_idx] + p1[cross_idx:]

        return offspring1, offspring2

    def mutate(self, chromosome):
        index =  randint(0,self.chromosome_length-1)
        # Swap the values at the selected indexes
        chromosome[index] = 1 - chromosome[index]

    def fitness(self, chromosome):
        length = len(self.required_services_per_machine)
        services_done_per_machine = [0]*length
        production_total = 0
        for i in range(len(chromosome)):
            if(chromosome[i] == 0):
                production_total += self.widgets_per_machine[i%length]
            else:
                services_done_per_machine[i%length] += 1 
        #Applying penalties if over serviced or under serviced
        for i in range(len(services_done_per_machine)):
            penalty = abs(self.required_services_per_machine[i] - services_done_per_machine[i])
            production_total -= 10*penalty
        return production_total

    def run(self):
        return super().run()


machines = [i for i in range(7)] # 10 cities
widgets_per_machine = [20,15,35,40,15,15,10]
required_services_per_machine = [2,3,1,1,2,1,1]
period = 4
chromosome_length = len(machines)*period
myMSP = MachineServicingProblem(chromosome_length, 100, 40, 0.9, 0.1, machines, widgets_per_machine,required_services_per_machine)
myMSP.run()

