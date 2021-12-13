import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt

class OxidizablePrecursors():
    def __init__(self, fname, precursor_labels):
        logprec = np.load(f'{fname}.npy')
        prec = 10**logprec
        self.prec = np.append(prec, np.sum(prec, axis = 1).reshape(len(prec), 1), axis = 1)
        self.logprec = logprec
        self.len = len(logprec)

        assert len(precursor_labels) == logprec.shape[1], f'Specifify {logprec.shape[0]} precursor labels'
        self.precursor_labels = precursor_labels + ['Total precursors']

    def trace(self, label, ax):
        """
        Plot the posterior trace

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = np.where(np.array(self.precursor_labels) == label)[0][0]
        ax.plot(range(self.len), self.logprec[:, idx], color = 'k')

    def mean(self):
        """
        Calculate the mean estimate of each precursor

            Returns:
                (array of floats) : mean of each precursor and total precursors
        """
        return(np.mean(self.prec, axis = 0))

    def quantile(self, quantile):
        """
        Calculate the quantile of each precursor

            Parameters:
                quantile (float) : A number between 0 and 1 exclusive
            Returns:
                (array of floats) : quantile estimate of each precursor and total precusors
        """
        assert 0 < quantile < 1, f'Input a quantile between 0 and 1'
        return(np.quantile(self.prec, quantile, axis = 0))

    def histogram(self, label, ax):
        """
        Plot the histogram of posterior

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = np.where(np.array(self.precursor_labels) == label)[0][0]
        ax.hist(self.logprec[:, idx])

    def kde(self, label, ax, color, linestyle):
        """
        Plot the kernel density estimator

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
                color : line color for kernel density
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = np.where(np.array(self.precursor_labels) == label)[0][0]
        sns.kdeplot(self.logprec[:, idx], color = color, linestyle = linestyle, label = label)

    def boxplot(self, ax, confidence_interval):
        """
        Plot the kernel density estimator

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
                confidence_interval (tuple) : whisker bounds
        """
        ax.boxplot(self.prec, whis = confidence_interval, showfliers = False)

    def posterior_predictive(self, top_delta, n_posterior, infered_columns, measured_pfca):
        """
        Plot posterior predictive against the measured delta of the TOP assay

            Parameters:
                top_delta (array) : measured delta in the TOP assay
                n_posterior (int) : number of samples from the posterior to compare
                infered_columns (list) : column indices of inferred columns (see makeA.py for indices)
                measured_pfca (list) : row indices of the measured PFCA (see makeA.py for indices)
            Returns:
                fig, ax (matplotlib figure) : figure of the posterior predictive
        """
        from makeA import makeA, x_ft, err_ft, x_ecf, err_ecf

        A, U = makeA(x_ft, err_ft, x_ecf, err_ecf)
        deleted_columns = []
        for col in range(A.shape[1]):
            if col not in infered_columns:
                deleted_columns.append(col)
        A = np.delete(A, deleted_columns, axis = 1)
        deleted_rows = []
        for row in range(A.shape[0]):
            if row not in measured_pfca:
                deleted_rows.append(row)
        A = np.delete(A, deleted_rows, axis = 0)

        random_indices = random.sample(range(self.len), n_posterior)
        post_predictive = []
        for i in random_indices:
            post_predictive.append(np.dot(A, self.prec[i, :-1]))

        fig, ax = plt.subplots()
        for idx, pp in enumerate(post_predictive):
            if idx == 0:
                ax.scatter(range(len(top_delta)), pp, label = 'posterior predictive', color = 'k', alpha = 0.5)
            else:
                ax.scatter(range(len(top_delta)), pp, color = 'k', alpha = 0.5)
        ax.scatter(range(len(top_delta)), top_delta, label = 'measured delta', color = 'red')
        ax.set_xticks(range(len(top_delta)))
        ax.set_ylabel('Concentration')
        ax.legend()
        return(fig, ax)

    def fluorotelomer_fraction(self):
        """
        Calculate the fraction of precursors that are fluorotelomers

            Returns:
                (tuple) : fluorotelomer fraction mean and standard deviation
        """
        fluorotelomer_concentration = np.sum(self.prec[:,['FT' in p for p in self.precursor_labels]], axis = 1)
        total_concentration = np.sum(self.prec, axis = 1)
        ft_fraction = fluorotelomer_concentration / total_concentration
        return(np.mean(ft_fraction), np.std(ft_fraction))

    def __organofluorine(self, n_F):
        """
        Convert from molar units to fluorine equivalents

            Paramets:
                n_F (array) : number of fluorines for each precursor
            Returns:
                (array) : precursor concentrations in fluorine equivalents
        """
        individual_precursors = self.prec[:, :-1]
        assert len(n_F) == individual_precursors.shape[1], 'specify the number of fluorines for each precursor'
        precursor_F = individual_precursors * n_F
        return(precursor_F)

    def eof_mean(self, n_F):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
            Returns:
                (float) : mean precursor concentration in fluorine equivalents
        """
        prec_F = self.__organofluorine(n_F)
        return(np.mean(np.sum(prec_F, axis = 1)))

    def eof_stdev(self, n_F):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
            Returns:
                (float) : standard deviation of precursor concentration in fluorine equivalents
        """
        prec_F = self.__organofluorine(n_F)
        return(np.std(np.sum(prec_F, axis = 1)))

    def eof_quantile(self, n_F, quantile):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
                quantile (float) : A number between 0 and 1 exclusive
            Returns:
                (float) : standard deviation of precursor concentration in fluorine equivalents
        """
        assert 0 < quantile < 1, f'Input a quantile between 0 and 1'
        prec_F = self.__organofluorine(n_F)
        return(np.quantile(np.sum(prec_F, axis = 1), quantile))
