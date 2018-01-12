import graph_data_construction
import pandas as pd
import numpy as np
#COMMAND TO GET DATA FROM SQL TABLES:
#SELECT sentenceId, userId, reaction, test.articles.Id as articleId FROM test.votes JOIN test.sentences on test.votes.sentenceId=test.sentences.id JOIN test.articles ON test.sentences.articleId = test.articles.id JOIN test.titles ON test.articles.id=test.titles.articleId


class OpinionBubbleTest:
    def __init__(self, csv_input = True, article_id = 40,
            input_data = '/Users/siddharth/flipsideML/ML-research/visualization/prodvotedata.csv',
            sample_size = 1.0, min_votes_user = 3, seed = 42):
        np.random.seed(seed)
        if csv_input:
            self.data = pd.read_csv(input_data)
        else:
            self.data = input_data
        self.sample_size = sample_size
        self.article_id = article_id
        self.model = None
        self.min_votes_user = min_votes_user

    def run(self):
        self.data = self.parse_data(self.article_id)[self.article_id]
        self.run_model()
        self.check_membership()
        assert len(self.viz_data) != 0, "didn't get out data"
        assert len(self.uncounted_users) == 0, "final users didn't work properly"
        assert len(self.ungrouped_users) == 0, "grouping didn't work properly"
        print("it worked wooooooo")

    def parse_data(self, articleId):
        out = {}
        c= 0
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            row = {key: row[key] for key in row.index}
            row['reaction'] = int(row['reaction'])
            if 'articleId' not in row:
                print(row)
                c += 1
            else:
                if row['articleId'] == articleId:
                    vote = {'userId': row['userId'], 'sentenceId': row['sentenceId'], 'reaction':row['reaction'], 'articleId': row['articleId']}
                    if row['articleId'] not in out.keys():
                        out[row['articleId']] = [vote]
                    else:
                        out[row['articleId']].append(vote)
                else:

                    vote = {'userId': row['userId'], 'sentenceId': row['sentenceId'], 'reaction':row['reaction']}
                    if row['articleId'] not in out.keys():
                        out[row['articleId']] = [vote]
                    else:
                        out[row['articleId']].append(vote)
        return out

    def run_model(self, votes = None, sample = 1.0):
        if votes is None:
            votes = self.data
        num_samples = None
        if type(sample) == float:
            num_samples = int(len(votes) * sample)
        else:
            num_samples = sample
        self.sampled_votes = np.random.choice(votes, num_samples, replace=False)
        self.model = graph_data_construction.Spectrum(min_votes_user = self.min_votes_user)
        self.model.add_votes(self.sampled_votes)
        self.viz_data = self.model.get_visualization_data()

    def check_membership(self):
        votes = self.sampled_votes
        viz_data = self.viz_data
        self.num_votes = {}
        self.questions = {}
        for vote in votes:
            if vote['userId'] not in self.num_votes.keys():
                self.num_votes[vote['userId']] = {'count':1, 'grouped':False}
            else:
                self.num_votes[vote['userId']]['count'] += 1
            if vote['sentenceId'] not in self.questions.keys():
                self.questions[vote['sentenceId']] = set([vote['userId']])
            else:
                self.questions[vote['sentenceId']].add(vote['userId'])

        self.uncounted_users = set()
        self.ungrouped_users = set()
        for group in viz_data:
            if group['group'] == 0:
                #ALL GROUP
                for user in self.num_votes.keys():
                    if user not in group['users']:
                        print('USER: {} NOT IN ALL GROUP'.format(user))
                        self.uncounted_users.add(user)
            else:

                for user in self.num_votes.keys():
                    if user in group['users']:
                        self.num_votes[user]['grouped'] = True
        enough_users = len([user for user in self.num_votes.keys() \
                            if self.num_votes[user]['count'] >= self.min_votes_user]) \
                            >= self.model.min_users
        voted_questions = [q for q in self.questions.keys() if len(self.questions[q]) >= self.model.min_votes_question]
        enough_questions = len(voted_questions) >= self.model.num_opinions
        if enough_users and enough_questions:
            for user in self.num_votes.keys():
                user_obj = self.num_votes[user]
                if (user_obj['grouped'] == False) and (user_obj['count'] > self.min_votes_user):
                    print('USER {} WITH {} VOTES NOT GROUPED'.format(user, user_obj['count']))
                    self.ungrouped_users.add(user)


TESTING_PARAMS = {"sample_size": [0.1, 0.5, 0.9, 1.0], "min_votes": [2, 3, 4, 5,6, 7, 8]}

if __name__ == "__main__":
    for size in TESTING_PARAMS["sample_size"]:
        for minim in TESTING_PARAMS["min_votes"]:
            try:
                print("RUNNING TEST FOR SIZE:{} MINIMUM VOTES/USER: {}".format(size, minim))
                test = OpinionBubbleTest(sample_size=size, min_votes_user = minim)
                test.run()
            except:
                print('failed on test with size: {} and minim: {}'.format(size, minim))
