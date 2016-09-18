import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from urllib2 import urlopen, Request
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import skfuzzy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,dendrogram


URL = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def scrape_list(site=URL):

    """
    scrape a list of current stocks in sp500 from wikipedia
    """

    tickers = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = Request(site, headers=headers)
    html = urlopen(req)
    soup = BeautifulSoup(html,"lxml")

    table = soup.find('table', {'class': 'wikitable sortable'})
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            ticker = str(col[0].string.strip())
            tickers.append(ticker)    
            
    return tickers

def download_data(tickers):

    """
    1. download the share price of the tickers from yahoo through python pandas
    2. calculate average daily log return and volatility
    """

    avg_returns = []
    vols = []
    tickers_downloaded = []
    for ticker in tickers:
        try:
            # download data from yahoo through python pandas
            ticker_data = web.DataReader(ticker,'yahoo',datetime.now() - relativedelta(days=30),datetime.now())

            # calculate avgerage daily log return
            avg_returns.append(np.mean(np.log(ticker_data['Adj Close']) - np.log(ticker_data['Adj Close'].shift(1))))
            
            # calculate volatility
            vols.append(np.std(np.log(ticker_data['Adj Close']) - np.log(ticker_data['Adj Close'].shift(1))))
            
            # when successful download, append the ticker to the list
            tickers_downloaded.append(ticker)
        except IOError as err:
            print ticker, ' got an error'
            pass

    print '{} tickers were downloaded'.format(len(tickers_downloaded))

    stock_metrics = pd.DataFrame({'avg_return':avg_returns,'volatility':vols},index=tickers_downloaded)
    return stock_metrics

def store_HDF5(stock_metrics):

    """
    store the calculated data in hdf5 file
    """

    store = pd.HDFStore('stock_clustering.h5')
    store['downloadDate{}'.format(datetime.now().strftime('%Y%m%d'))] = stock_metrics
    store.close()

def get_sp500_data():
    tickers = scrape_list()
    stock_metrics = download_data(tickers)
    store_HDF5(stock_metrics)



class FuzzyCMeans(object):
    """
    This class apply fuzzy c means to the stock data
    """

    def __init__(self,download_date = datetime.now().strftime("%Y%m%d")):
        """
        Pull the avg. return and volativlity of sp500 stocks from hdf5 file
        """
        # retrieve data
        store = pd.HDFStore('stock_clustering.h5')
        df = store['downloadDate{}'.format(download_date)]

        # save avg. returns and volatilities as one stacked vector
        self.data = np.vstack((df['avg_return'].values,df['volatility'].values))

        store.close()


    def visualize_data(self):

        """
        draw a scatter plot of original data with avg. daily returns as x-axis
        and volatilities as y-axis
        """
        plt.scatter(self.data[0], self.data[1] ,alpha=0.6)
        plt.xlabel('avarage daily return')
        plt.ylabel('volatility')
        plt.title('sp500 stocks: past 30days')
        plt.savefig('scatter_plot_original_data.png',dpi=300)
        
        
    def find_clusters(self,min_num=2, max_num=6):

        """
        apply different numbers of clusters and find the optimal number of clusters
        """

        # save the min and max of clusters to try as attribute
        self.min_num = min_num
        self.max_num = max_num

        #options for colors in the plot
        colors= ['b', 'g', 'r', 'c', 'm', 'y', 'k','orange', 'Brown', 'ForestGreen']

        # In the plot, fix the number of columns at 3
        # adjust the number of rows in the frame to contain all the plots
        ncols = 3
        nrows = max_num // ncols if max_num % ncols == 0 else max_num // ncols + 1

        fig1, ax1 = plt.subplots(nrows, ncols,figsize=(8,8))
        self.fpcs = [] # This is going to be the list of fuzzy partition coefficients

        for num_clusters, ax in enumerate(ax1.reshape(-1),2):
            center, u, u0, d,jm, p, fpc = skfuzzy.cluster.cmeans(self.data, num_clusters, 2,\
             error=0.005,maxiter=1000,init=None)

            # save fuzzy partition coefficients at each number of clusters
            self.fpcs.append(fpc)

            # assign cluster membership of the highest probability 
            cluster_membership = np.argmax(u, axis=0)

            # draw a color-coded plot
            for i in range(num_clusters):
                ax.plot(self.data[0][cluster_membership == i], \
                    self.data[1][cluster_membership == i],'.',color=colors[i])

            # place a marker at each center of the clusters
            for loc in center:
                ax.plot(loc[0],loc[1],'rs')

            ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(num_clusters,fpc))
            ax.axis('off')

        fig1.tight_layout()
        plt.savefig('fuzzy_cmeans_cluster_selection.png')

    def draw_fpcs(self):

        """
        draw a fuzzy partition coefficients against numbers of clusters
        The optimail choice is the heighest one
        """

        fig, ax = plt.subplots()
        ax.plot(np.r_[self.min_num:(self.max_num + 1)],self.fpcs[:(self.max_num - self.min_num + 1)])
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Fuzzy Partition Coefficient")
        plt.savefig('fuzzy_partition_coefficient_plot.png',dpi=300)

        plt.cla()
        plt.clf()

    def soft_assign(self, nclusters=2):

        """
        assign a cluster to each data point. A cluster membership is determined by the probability
        of belonging to the cluster.
        Those data points with low probability has faded colors.
        """
        self.nclusters = nclusters

        # run fuzzy cmeans with fixed number of cluster
        center,u,_,_,_,_,_ = skfuzzy.cluster.cmeans(self.data,nclusters,2,error=0.005,maxiter=1000,init=None)

        # determine cluster membership
        cluster_membership = np.argmax(u,axis=0)

        # get the heightest probability for each data point
        probabilities = np.amax(u,axis=0)

        # rgba_colors matrix controls the colors and fade of cluster members
        rgba_colors = np.zeros((len(self.data[1]),4))
        for row, col in enumerate(cluster_membership):
            rgba_colors[row,col] = 1.0

        # the 4th column of rgba_colors controls fading. 
        # to pronnouce the differences in probabilitis, cube each value
        rgba_colors[:,3] = [i**3 for i in probabilities]



        plt.scatter(self.data[0],self.data[1],color=rgba_colors)
        plt.savefig('fuzzy_cmeans_soft_assignment.png')
        


class HCluster(object):
    """
    This class apply hierarchical clustering, specifically agglomerative hierarchical clustering
    """

    def __init__(self,download_date=datetime.now().strftime("%Y%m%d")):
        """
        pull data from hdf5 file and save data and ticker names as attributes
        """
        store = pd.HDFStore('stock_clustering.h5')
        self.df = store['downloadDate{}'.format(download_date)]
        self.labels = self.df.index.values

        store.close()


    def agglomerative_clusters(self,metric='euclidean',method='complete'):
        """
        Apply agglomerative hierarchical clustering
        """
        #create clusters with scipy module linkage and pdist
        clusters = linkage(pdist(self.df, metric=metric), method=method)

        # draw a dendrogram
        dendr = dendrogram(clusters,labels=self.labels)
        plt.ylabel('Euclidean distance')
        plt.tight_layout()
        plt.savefig('hierarchical_clustering.png',dpi=300,bbox_inches='tight')
        
        


if __name__ == "__main__":
    get_sp500_data()

    fcm = FuzzyCMeans()
    fcm.visualize_data()
    fcm.find_clusters()
    fcm.draw_fpcs()
    fcm.soft_assign()

    hc = HCluster()
    hc.agglomerative_clusters()

    



