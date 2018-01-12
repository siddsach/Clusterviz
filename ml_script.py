import graph_data_construction
import sys, json, numpy


#QUESTIONS = ['q1','q2','q3','q4','q5','q6','q7','q8']
#Read data from stdin

def read_inp():
	raw_inp = sys.stdin.readlines()
	inp = json.loads(raw_inp[0])[0]
	return inp

def main():
    votes = read_inp()

    model = graph_data_construction.Spectrum()#QUESTIONS)

    model.add_votes(votes)
    out_data = model.get_visualization_data()
    print(json.dumps(out_data))

#start process
if __name__ == '__main__':
    main()
