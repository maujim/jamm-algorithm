import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from ortools.linear_solver import pywraplp
import pulp as lp

import time
import sys
import re
import itertools
import logging
import datetime

logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logging.warning("You are learning Python logging!")


file_path = "./arxiv_data.csv"
df = pd.read_csv(file_path)

logging.warning("done reading csv data into df")

# ======= data preprocess =======
# print all 'terms'
pat = re.compile(r"'(\S*?)'")

df["terms_clean"] = [re.findall(pat, tt) for tt in df.terms]

unique_terms = set(x for term in df.terms_clean for x in term)
# print(unique_terms)

# for now, keep only computer vision papers
df = df.drop(df.index[~df["terms_clean"].apply(lambda x: "cs.CV" in x)].tolist())

# only keep 10k papers to run shit faster
max_papers = 1000
df = df.head(max_papers)

logging.warning(f"cleaned data, keeping only {max_papers} papers with cs.CV")


# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the summaries to a document-term matrix
dtm = vectorizer.fit_transform(df["summaries"])

logging.warning("finished building dtm")

# TODO: maybe save dtm to a csv file so i don't have to do these steps everytime

# ======= do the LSA =======

# Initialize the TruncatedSVD with the desired number of components
lsa = TruncatedSVD(
    n_components=500,  # Adjust this as needed
)

# Fit the LSA model to the document-term matrix
lsa.fit(dtm)

# Transform the document-term matrix to the lower-dimensional space
dtm_lsa = lsa.transform(dtm)

logging.warning(
    f"went from original DTM shape of {dtm.shape} to LSA transformed DTM shape of {dtm_lsa.shape}"
)

# ======= get pairwise cosine distances =======

cosine_dist_matrix = cosine_distances(dtm_lsa)

logging.warning("created cosine_dist_matrix")

# ======= use ortools instead =======
#
# TODO: try using the code linked in the neuromatch paper itself
# https://github.com/titipata/paper-reviewer-matcher/blob/0fc8b04081d391d2816b911d6839cddc559e9dc4/paper_reviewer_matcher/lp.py

num_attendees = len(cosine_dist_matrix)

# Initialize the linear solver
solver = pywraplp.Solver.CreateSolver("SCIP")

# Define binary decision variables for matching attendees
matches = {}
for i in range(num_attendees):
    for j in range(num_attendees):
        matches[i, j] = solver.BoolVar(f"Match_{i}_{j}")

# Define the objective function: minimize total cosine distance
objective = solver.Objective()
for i in range(num_attendees):
    for j in range(num_attendees):
        objective.SetCoefficient(matches[i, j], cosine_dist_matrix[i][j])
objective.SetMinimization()

# Constraint: Each person should have at most 3 matches
for i in range(num_attendees):
    solver.Add(sum(matches[i, j] for j in range(num_attendees)) <= 3)

# Solve the linear programming problem
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Solution:")
    for i in range(num_attendees):
        for j in range(num_attendees):
            soln_value = matches[i, j].solution_value()
            if soln_value > 0.5:
                print(
                    f"Attendee {i+1} matches with Attendee {j+1} with value {soln_value}"
                )
else:
    print("No optimal solution found.")

sys.exit(0)

# ======= time for the linear programming =======

# Assuming cosine_dist_matrix comes from scipy's cosine_distances(dtm_lsa)
# Number of attendees and matches per person
num_attendees = cosine_dist_matrix.shape[0]
matches_per_person = 3  # Adjust as needed

# Define the linear programming problem
prob = lp.LpProblem("Optimal Matching Problem", lp.LpMinimize)

# Define the assignment matrix as a binary variable
assign_matrix = lp.LpVariable.dicts(
    "Assign",
    [(i, j) for i in range(num_attendees) for j in range(num_attendees)],
    cat="Binary",
)

# Objective function: minimize total cosine distance
prob += lp.lpSum(
    [
        cosine_dist_matrix[i, j] * assign_matrix[(i, j)]
        for i in range(num_attendees)
        for j in range(num_attendees)
    ]
)

# Constraints: each attendee should have at most matches_per_person matches
for i in range(num_attendees):
    prob += (
        lp.lpSum([assign_matrix[(i, j)] for j in range(num_attendees)])
        <= matches_per_person
    )

# Solve the linear programming problem
prob.solve()

# Extract the optimal assignment matrix
optimal_assign_matrix = np.zeros((num_attendees, num_attendees), dtype=int)
for v in prob.variables():
    if v.varValue == 1:
        i, j = v.name.split("_")[1][1:-1].split(",")
        optimal_assign_matrix[int(i), int(j)] = 1

# Print the optimal assignment matrix
print("Optimal Assignment Matrix:")
print(optimal_assign_matrix)
