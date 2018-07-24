import numpy as np
import pandas as pd
from tabulate import tabulate

envs=["Taxi", "Loop", "NChain-v0"]
policies=["Boot", "Weighted"]
num_algs=len(policies)

for env in envs:
    tableData={"Algorithm":[""]*num_algs, "Avg Score Phase 1":[1.]*num_algs, "Std Dev Phase 1":[1.]*num_algs ,"Avg Score Phase 2":[1.]*num_algs,"Std Dev Phase 2":[1.]*num_algs}
    count=0
    for policy in policies:
            path=env+"/r_"+policy+".npy"
            path_test=env+"/r_test_"+policy+".npy"
            try:
                scores=np.load(path)
                scores_test=np.load(path_test)
                tableData["Avg Score Phase 1"] [count], tableData["Std Dev Phase 1"][count]=np.mean(np.sum(scores, axis=1)), np.std(np.sum(scores, axis=1))
                tableData["Avg Score Phase 2"] [count], tableData["Std Dev Phase 2"][count]=np.mean(np.sum(scores_test, axis=1)), np.std(np.sum(scores_test, axis=1))
                tableData["Algorithm"][count]=policy
                count+=1
            except IOError:
                print(path+" not found")
    df=pd.DataFrame(tableData)
    print (tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv(env+"/aggregates.csv", sep=',')
