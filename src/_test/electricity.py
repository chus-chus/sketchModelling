import skmultiflow as skm

# todo apply means, target as onehot

if __name__ == "__main__":
    # Sudden drift clearly separated, NB, nwait 50
    stream = skm.data.FileStream('./data/rawStreams/electricity.csv')
    evaluator = skm.evaluation.EvaluatePrequential(n_wait=50, show_plot=False, pretrain_size=200, max_samples=20000,
                                                   output_file='./logs/sine/NB_EHmeansSuddenDriftSeparatedResults.csv')
    evaluator.evaluate(stream=stream, model=skm.bayes.NaiveBayes())