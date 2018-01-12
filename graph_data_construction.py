import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score, pairwise
import json
import sys

'''
class to process (userId, sentenceId, reaction) votes to cluster users and calculate statistics for visualization
'''


class Spectrum:
    def __init__(self, graph_low_votes = False, reducer = 'mds', cluster = 'kmeans',
        choosing_function = 'diff', norm_threshold = 100, defval = 0, impute_factor = True,
        min_users = 10, min_votes_user = 3, min_votes_question = 3, num_opinions = 3,
        max_users = 1000, n_components = 2, min_votes_group = 2):

        self.impute_factor = impute_factor
        self.min_votes_user = min_votes_user
        self.min_votes_question = min_votes_question
        self.min_votes_group = min_votes_group
        self.min_users = min_users
        self.graph_low_votes = graph_low_votes


        self.n_components = n_components

        self.max_users = max_users

        if reducer == 'mds':
            self.dimension_reducer = MDS
        elif reducer == 'pca':
            self.dimension_reducer = PCA

        if cluster == 'affinity':
            self.cluster = AffinityPropagation
        elif cluster == 'kmeans':
            self.cluster = KMeans

        self.num_opinions = num_opinions

        self.choosing_function = choosing_function
        self.norm_threshold = norm_threshold

        self.data = np.zeros((self.max_users, 100))

        self.users = dict()
        self.n_users = 0
        self.users_to_graph = list()
        self.votes_to_consider = dict()
        self.raw_votes = dict()
        self.normalize = False
        self.group_averages = None

        self.question_ids = []
        self.question_votes = dict()

        self.k = None
        self.considered_points = None
        self.out_points = None
        self.groups = None
        self.relevant_claims= None


    def save_model(self):

        out = dict()
        out['questions'] = {'n_questions':len(self.question_ids), 'questions': list(self.question_ids)}

        vote_matrix = []
        for i in range(self.n_users):
            vote_matrix.append([float(x) for x in list(self.data[i])])
        out['votes'] = {'n_users': self.n_users, 'matrix': vote_matrix}

        out['user_ids'] = self.users
        out['users_to_graph'] = self.users_to_graph
        out['k'] = self.k

        out['out_points'] = []
        if self.out_points is not None:
            for i in range(len(self.out_points)):
                x, y, group = self.out_points[i,0], self.out_points[i,1], self.groups[i]
                out['out_points'].append({"x": float(x), "y": float(y), "group": int(group)})

        out['relevant_questions'] = self.relevant_claims

        return json.dumps(out)

    def reinit_model(self, data):
        self.question_ids = np.array(data['questions'])
        self.n_users = data['votes']['n_users']
        for i in range(len(data['votes']['matrix'])):
            self.data[i] = data['votes']['matrix'][i]
        self.users = data['user_ids']
        self.users_to_graph = data['users_to_graph']
        self.k = data['k']
        self.out_points = np.zeros((len(data['out_points']), 2))
        self.groups = np.zeros(len(data['out_points']))
        for i in range(len(data['out_points'])):
            dic = data['out_points'][i]
            self.out_points[i][0], self.out_points[i][1] = np.float64(dic['x']),np.float64(dic['y'])
            self.groups[i] = np.int32(dic['group'])
        self.relevant_questions = {int(k): v for k,v in data['relevant_questions'].items()}





#CLASS INTERFACE TO PROCESS VOTES HERE, ASSUMES VOTE IS (str) 1 if agree, 2 if disagree, 3 if pass
#Input: list of 3-tuples of form (user_id, question_id, vote)
    def add_votes(self, input_votes, process = True):
        #enough_votes = False DONT NEED THIS LINE BECAUSE SCRIPT IS RUNNING EVERY TIME
        votes = [(el['userId'], el['sentenceId'], el['reaction']) for el in input_votes]
        for user_id, question_id, vote in votes:
            assert (vote == 1 or vote == 0 or vote == -1)
            if user_id not in self.users.keys():
                self.users[user_id] = {"index":self.n_users, "questions": set([question_id])}
                self.n_users += 1
            else:
                self.users[user_id]["questions"].add(question_id)

            if question_id not in self.raw_votes.keys():
                self.raw_votes[question_id] = set([(user_id, vote)])
            else:
                self.raw_votes[question_id].add((user_id, vote))

            if question_id not in self.votes_to_consider.keys():
                self.votes_to_consider[question_id] = set([user_id])
            else:
                self.votes_to_consider[question_id].add(user_id)

            self.add_question(question_id)

            self.add_vote(user_id, question_id, vote)

        enough_users = (len(self.users_to_graph) >= self.min_users)
        enough_questions = len(self.question_ids) >= self.num_opinions
        if process and enough_users and enough_questions:
            self.process()


    def add_vote(self, user_id, question_id, vote):
        user_ind = self.users[user_id]["index"]
        question_ind = self.get_question_index(question_id)
        if question_ind is not None:
            self.data[user_ind][question_ind] = vote

        #regardless of if the questions a user has voted on have been voted enough
        #to be counted, they will be added to users to graph if they've voted
        # a certain number of times
        if len(self.users[user_id]["questions"]) >= self.min_votes_user:
            #add index of user if they've added enough votes
            if self.users[user_id]["index"] not in self.users_to_graph:
                self.users_to_graph.append(self.users[user_id]["index"])

    def add_question(self, question_id):
        if question_id not in self.question_ids:
            if len(self.votes_to_consider[question_id]) >= self.min_votes_question:
                self.question_ids.append(question_id)
                for user_id, vote in self.raw_votes[question_id]:

                    self.add_vote(user_id, question_id, vote)

                if len(self.question_ids) >= self.data.shape[1]:
                    new = np.zeros((self.max_users, len(self.question_ids) + 20))
                    new[:self.data.shape[0], :self.data.shape[1]] = self.data
                    self.data = new

    def get_question_index(self, question_id):
        questions = np.array(self.question_ids)
        if question_id in questions:
            index = np.where(questions == question_id)[0][0]

            return index
        else:
            return None



    '''

    output of following form
    var clusterData= [
      {cluster: "A", opinions: [{sentenceId:"s1q1", decision: "agree", average: 100}, {sentenceId:"s1q1", decision: "agree", average: 100}]}
    ];
    //111 and 112 represent userIds
    var pointData = {111: {cluster: "A", "x": 100, "y": 50}, 112:{cluster: "A", "x": 100, "y": 50}}:
    var shadeData = [
      {cluster: "A", shading:[{ "x": 0, "y": 0},  { "x": 100,  "y": 0}, { "x": 100,  "y": 200}, { "x": 0,   "y": 0}]},
      {cluster: "B", shading:[{ "x": -10,   "y": -10},  { "x": -30,  "y": -150}, { "x": -200,  "y": -250}, { "x": -10,   "y": -10}]}
    ];
    output = {"clusterData": clusterData, "pointData":pointData, "shadeData": shadeData}
    '''


    def get_visualization_data(self):
        if len(self.votes_to_consider) != 0:
            data = []
            index_by_ind = {v["index"]:k for k,v in self.users.items()}
            user_ids = [index_by_ind[ind] for ind in self.users_to_graph]

            #Adding group placeholder for people who aren't considered yet
            clusters = self.groups is not None
            r = 0
            if self.k is not None:
                r = self.k
            for i in range(-1, r):
                #For all users group, i will be -1 and group index will be 0
                if not clusters:
                    if i > -1:
                        return data
                group = dict()
                group['group'] = i + 1
                users = None
                gs = None
                if self.groups is not None:
                    gs = list(self.groups)
                else:
                    gs = [None] * len(user_ids)
                if i != -1:
                    users = []
                    for iden, g in zip(user_ids, gs):
                        #Check if the user is in the group or if we're considering all users
                        if g == i:
                            users.append(iden)
                else:
                    users = set()
                    for subset in self.votes_to_consider.values():
                        users.update(subset)
                    users = list(users)

                group['users'] = users
                group['size'] = len(users)

                if self.relevant_claims is None:
                    self.relevant_claims = {-1: set(self.question_ids)}
                relevant_positions = []
                for question in self.relevant_claims[i]:
                    claim_data = dict()
                    claim_data['sentenceId'] = str(question)
                    avg,controversiality, num_votes,proportions = self.get_numbers(i, question)
                    if avg is not None:
                        avg = float(avg)
                    if controversiality is not None:
                        controversiality = float(controversiality)
                    if num_votes is not None:
                        num_votes = int(num_votes)
                    claim_data['average'] = avg
                    claim_data['shadeColor'] = self.range_normalize(0, 1, 0.3, 0.1, avg)
                    claim_data['controversiality'] = controversiality
                    claim_data['num_votes'] = num_votes
                    for answer, direction in zip([-1,0,1], ['disagree', 'unsure', 'agree']):
                        if len(proportions) != 0:
                            claim_data[direction] = float(proportions[answer])
                        else:
                            claim_data[direction] = None
                    relevant_positions.append(claim_data)
                group['sentences'] = relevant_positions
                data.append(group)
                del group
            return data
        else:
            return []

    def range_normalize(self, minimum, maximum, newmin, newmax, value):
        if value == 0:
            return 0
        elif value is not None:
            sign = value > 0
            factor = newmax - newmin
            denom = maximum - minimum
            out = (factor * ((value - minimum) / denom)) + newmin
            if sign:
                return out
            else:
                return -out
        else:
            return None

    def get_numbers(self, i, question):
        votes = []
        seen_users = set()
        for user_id, vote in self.raw_votes[question]:
            if i == -1:
                if user_id not in seen_users:
                    votes.append(vote)
                    seen_users.add(user_id)
            else:
                if user_id not in seen_users:
                    user_ind = self.users[user_id]['index']
                    if user_ind in self.users_to_graph:
                        index = np.where(np.array(self.users_to_graph) == user_ind)[0][0]
                        if self.groups[index] == i:
                            votes.append(vote)
                            seen_users.add(user_id)

        group_answers = {}
        avg = None
        controversiality = None
        num_votes = len(votes)
        if num_votes >= self.min_votes_group:

            controversiality = float(np.std(votes) / np.sqrt(num_votes))
            avg = float(np.mean(votes))
            votes = np.array(votes)
            for answer in [-1,0,1]:
                n_votes = float(np.where(votes == answer)[0].shape[0])
                group_answers[answer] = float(n_votes / num_votes)

        return avg,controversiality,  num_votes, group_answers

    def get_agreement_phrase(self, i, question):
        value = self.average_answer(i, question)
        opinion = None
        if value > 0:
            opinion = 'agreement'
        elif value < 0:
            opinion = 'disagreement'
        elif value == 0:
            opinion = 'neutral'

        degree = None
        if 0 <= abs(value) < 0.33:
            degree = 'weak'
        elif 0.33 <= abs(value) < 0.67:
            degree = 'moderate'
        elif 0.67 <= abs(value) <= 1:
            degree = 'strong'

        if opinion == 'neutral':
            return opinion
        else:
            return 'in ' + degree + ' ' + opinion

    def get_points(self):
        if self.out_points is not None:
            return self.out_points
        else:
            print('need more data')
            return None

    def get_groups(self):
        if self.groups is not None:
            return self.groups
        else:
            print('need more data')
            return None


    def convert(self, vote):
        if int(vote) == 1:
            return 1
        elif int(vote) == 2:
            return -1
        elif int(vote) == 3:
            return 0

    def process(self):
        if self.n_users >= self.norm_threshold:
            self.normalize = True

        self.dimension_reduction()
        self.make_groups()
        if self.groups is not None:
            if self.cluster == KMeans:
                self.find_relevant_claims()

    def impute(self, data):
        if self.impute_factor:
            filled_in = np.where(data != 0)
            done_votes = data[filled_in]

            #IMPUTING WITH SVD
            u , s, v = np.linalg.svd(data)

            #MAKE SINGULAR VALUE VECTOR DIAGONAL MATRIX
            diag = np.zeros(data.shape)
            diag[range(min(data.shape)), range(min(data.shape))] = s

            new_data = np.matrix(u) * np.matrix(diag) * np.matrix(v)

            new_data[filled_in] = done_votes
            return new_data
        else:
            return data

    def dissimilarity(self, X):
        return pairwise.cosine_distances(X)


    def dimension_reduction(self):
        new_data = self.data[self.users_to_graph]
        if self.impute_factor:
            new_data = self.impute(new_data)
        if self.dimension_reducer == PCA:
            self.dimension_reducer = self.dimension_reducer(n_components = self.n_components, metric = False)
            self.considered_points = self.dimension_reducer.fit_transform(new_data)
            self.out_points = self.dimension_reducer.transform(self.impute(self.data[:self.n_users]))
        elif self.dimension_reducer == MDS:
            self.dimension_reducer = self.dimension_reducer(n_components = self.n_components,
                    dissimilarity = 'precomputed', metric = False, n_init = 20, random_state = 0)
            self.considered_points = self.dimension_reducer.fit_transform(self.dissimilarity(new_data))

            #MDS cannot perform feature transformation on new data, so cannot graph people who don't have enough votes
            self.out_points = self.considered_points

    def make_groups(self):
        best_score = 0
        best_groupings = None
        best_k = None
        k = 2

        if self.cluster == KMeans:

            while k <= len(self.users_to_graph) // 2:
                kmeans = self.cluster(n_clusters = k, n_init = 50, random_state = 0)
                groups = kmeans.fit_predict(self.considered_points)
                num_members = [(groups == k_num).sum() for k_num in range(k)]

                if min(num_members) >= 3:
                    score = silhouette_score(self.considered_points, groups)
                    if score > best_score:
                        best_score = score
                        best_groupings = groups
                        best_k = k
                k += 1

            self.k = best_k
            self.groups = best_groupings

        elif self.cluster == AffinityPropagation:
            preference = [len(np.where(np.nonzero(self.data[self.users_to_graph])[0] == i)[0]) \
                                                    for i in range(len(self.users_to_graph))]
            self.cluster = self.cluster(preference = preference)
            self.groups = self.cluster.fit_predict(self.considered_points)
            self.k = len(self.cluster.cluster_centers_indices_)



    def find_stds(self):
        question_std = np.zeros(len(self.question_ids))
        for q_ind in range(len(self.question_ids)):
            u_inds = list(self.votes_to_consider[q_ind])
            stdev = np.std(self.data[u_inds, q_ind])
            if stdev == 0:
                #prevent from dividing by zero
                stdev = 0.000001
            question_std[q_ind] = stdev
        return question_std

    def get_group_averages(self):
        #GET AVG FOR EACH GROUPS
        # MUST DO FANCY INDEXING TO BOTH RESTRICT TO CERTAIN GROUPS AND TO IGNORE SPOTS THAT HAVENT
        #GOTTEN VOTES YET
        data = {}
        for c in range(self.k):
            #indexes according to data only with graphed users
            group_indices = np.where(self.groups == c)[0]
            #reindexing along original index for users by self.n_users which we need for self.votes_to_consider
            grouped_users = [self.users_to_graph[i] for i in group_indices]
            averages = []
            counts = []
            for q_id in self.question_ids:
                #GETTING USERS WHO HAVE VOTED ON q_id
                voted_users = set([self.users[user]["index"] \
                    for user in self.votes_to_consider[q_id]])
                users = list(set(grouped_users).intersection(voted_users))
                q_ind = self.get_question_index(q_id)
                count = len(users)
                counts.append(count)
                if count > 0:
                    averages.append(np.mean(self.data[users, q_ind]))
                else:
                    averages.append(0)
            data[c] = {"averages": np.array(averages), "counts": np.array(counts)}
        self.group_averages = data

    def differentiate_claims(self, question_std):
        relevant_questions = dict()
        all_questions = []
        for group in range(self.k):
            sum_squared_differences = np.zeros(len(self.question_ids))
            for other_group in range(self.k):
                if group != other_group:
                    mean_squared_deviations = (self.group_averages[group]["averages"] - \
                                                    self.group_averages[other_group]["averages"])**2
                    sum_squared_differences += mean_squared_deviations
            #Taking the opinions for which differences are biggest
            deviations = [(diff, q_id) for diff, q_id
                    in zip(sum_squared_differences, self.question_ids)]
            sorted_questions = sorted(deviations, reverse = True)
            #Ignoring questions for which deviation is zero
            sorted_questions = [el[1] for el in sorted_questions]
            relevant_questions[group] = sorted_questions
            all_questions += sorted_questions
        relevant_questions[-1] = set(all_questions)
        return relevant_questions

    def strongest_claims(self, question_std):
        all_opinions = []

        for group in self.group_averages.keys():
            #Indexed_averages --> [(group_avg, group, question),,...]
            indexed_averages = [(abs(self.group_averages[group]["averages"][q]), group, q)
                                    for q in range(len(self.group_averages[group]))
                                    if self.group_averages[group]["counts"][q] >= self.min_votes_group]
            all_opinions += indexed_averages

        all_opinions = sorted(all_opinions, reverse = True)

        strongest_claims = dict()
        strongest_claims[-1] = []
        for avg, question_ind, group in all_opinions:
            '''
            if question_ind not in question_candidates.keys():
                question_candidates[question_ind] = 1
            else:
                question_candidates[question_ind] += 1
            if question_candidates[question_ind] <= self.k // 2:
            '''
            if group in strongest_claims.keys():
                if len(strongest_claims[group]) >= self.num_opinions:
                    pass
                else:
                    strongest_claims[group].append(self.question_ids[question_ind])
            else:
                strongest_claims[group] = [self.question_ids[question_ind]]

        for i in range(self.k):
             strongest_claims[-1] += strongest_claims[i]
        strongest_claims[-1] = list(set(strongest_claims[-1]))
        return strongest_claims

    def find_relevant_claims(self, use_std = False):
        question_std = None
        if self.normalize:
            try:
                if use_std:
                    question_std = self.find_stds()
            except:
                sys.stderr.write("Couldn't Normalize")
                pass

        self.get_group_averages()

        if self.choosing_function == 'strong':
            chooser = self.strongest_claims
        elif self.choosing_function == 'diff':
            chooser = self.differentiate_claims

        self.relevant_claims = chooser(question_std)


# FIND OUTLINE OF CLUSTERING IN 2D SPACE FOR POLYGON VISUALIZATION

# Function to know if we have a CCW turn
def RightTurn(p1, p2, p3):
    if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
        return False
    return True

# Main algorithm:
def GrahamScan(P):
    P.sort()            # Sort the set of points
    L_upper = [P[0], P[1]]      # Initialize upper part
    # Compute the upper part of the hull
    for i in range(2,len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1],L_upper[-2],L_upper[-3]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]    # Initialize the lower part
    # Compute the lower part of the hull
    for i in range(len(P)-3,-1,-1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1],L_lower[-2],L_lower[-3]):
            del L_lower[-2]
    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower       # Build the full hull
    return np.array(L)





'''
if self.graph_low_votes:
data = {"clusterData" : [], "pointData" : {}, "shadeData":[]}
index_by_ind = {v[0]:k for k,v in self.users.items()}
user_ids = [index_by_ind[ind] for ind in range(self.n_users)]
xs, ys = self.out_points[:,0], self.out_points[:,1]
x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
data['extremes'] = {"xMin": float(x_min), "xMax": float(x_max), "yMin": float(y_min), "yMax": float(y_max)}
#Adding group placeholder for people who aren't considered yet
groups = []
c = 0
for i in range(self.n_users):
if i in self.users_to_graph:
    groups.append(self.groups[c])
    c += 1
else:
    groups.append(None)

for iden, x, y, group in zip(user_ids, xs, ys, groups):
group_label = None
if group is not None:
    group_label = int(group)
else:
    group_label = 'UNGROUPED: NOT ENOUGH VOTES'

data["pointData"][iden] = {"x":x, "y":y, "cluster":group_label}

for i in range(self.k):
group_data = []
c = 0
for iden, x, y, group in zip(user_ids, xs, ys, groups):
    group_label = None
    if group == i:
        group_data.append({"iden":iden, "x":x, "y":y, "group":int(group)})
        group_label = int(i)
    else:
        group_label = 'UNGROUPED: NOT ENOUGH VOTES'

points = [(el["x"],el["y"]) for el in group_data]
perimeter_points = GrahamScan([(el["x"],el["y"]) for el in group_data])
path = [{"x":x,"y":y} for x, y in zip(perimeter_points[:,0], perimeter_points[:,1])]
path.append(path[0])
data["shadeData"].append({"cluster":int(i), "shading": path})
if self.relevant_questions is not None:
    relevant_positions = [{"sentenceId":question,"cluster":int(i), "average":self.average_answer(i, question), "phrase": self.get_agreement_phrase(i,question)} for question in self.relevant_questions[i]]
    data["clusterData"].append(relevant_positions)
return data
'''


