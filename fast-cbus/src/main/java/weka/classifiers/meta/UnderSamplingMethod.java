package weka.classifiers.meta;

import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;

/*
 * This class performs undersampling
 */
public class UnderSamplingMethod implements Serializable {
	private int m; // parameter for calculating the desired ratio of the majority after the undersampling
	private int size_mi; // count size of minority. will be initialized at setMiniritySize()
	private long seed=-1; // for choosing majority elements from the cluster
	private int majority_index; // index of the majority class
		
	public UnderSamplingMethod(int m,long seed){
		this.m=m;
		this.seed=seed;
	}
	public UnderSamplingMethod(int m){
		this.m=m;
	}
	
	public Instances underSample(Instances data,Clusterer clusterer) throws Exception
	{
		int numOfClusters = data.attribute("cluster").numValues();
		
		this.majority_index = 1-ImbalancedUtils.findMinorityClass(data);
		this.size_mi = data.attributeStats(data.classIndex()).nominalCounts[1-this.majority_index];
		
		// Use the method described in the paper to reduce the majority samples
		// per cluster. Use the formula in the paper.
		Instances[] instancesPerCluster = ImbalancedUtils.getClustersInstances(data);
		double [] clustersRatio = getRatio(instancesPerCluster,numOfClusters);
		
		// Leave all minority instances
		Instances balancedData = new Instances(data);
		randomFiltering(balancedData,size_mi,1-majority_index);

		// add majority instances from each cluster
		int size_ma;
		for (int i=0;i<numOfClusters;i++){
			size_ma=getMajoritySizeInCluster(clustersRatio,i);	// size of majority to choose from current cluster
			randomFiltering(instancesPerCluster[i],size_ma,majority_index);
			balancedData.addAll(instancesPerCluster[i]);
		}
		return balancedData;
	}	
	
	/**
	 * Filters a given data set, leaving only instances of the desired class, and only the desired amount of instance.
	 * get the Majority/Minority class instances for the current Cluster, having the desired size
	 * @param data
	 * @param size_ma
	 */
	private void randomFiltering(Instances data,int size,int classIndex){
		//remove all other class instances
		Iterator<Instance> instanceIterator = data.iterator();
		Instance current;
		while(instanceIterator.hasNext()){
			current = instanceIterator.next();
			if (current.classValue() != classIndex)
				instanceIterator.remove();
		}

		// removing classIndex instances until desired size reached
		Random rnd;
		while(size<data.size()){
			if (seed!=-1)//provided with seed
				rnd = new Random(seed);
			else 
				rnd = new Random();
			data.remove(rnd.nextInt(data.size()));
		}
	}
	
	
	/**
	 * 	Calculate the number of wanted majority
	 * @param clusters_ratio
	 * @param cluster_index
	 * @return
	 */
	private int getMajoritySizeInCluster(double [] clusters_ratio,int cluster_index){
		int size_ma=0;
		double denominator=0;
		for (int i=0;i<clusters_ratio.length;i++){
			denominator+=clusters_ratio[i];
		}
		// calculated by: http://sci2s.ugr.es/keel/pdf/specific/articulo/yen_cluster_2009.pdf
		size_ma=(int) ( (m*size_mi)*(clusters_ratio[cluster_index]) / denominator);
		return size_ma;
	}
	
	/**
	 * for each cluster: majority/minority
	 * calculate ratio 
	 * @param instancesPerCluster
	 * @param numOfClusters
	 * @return
	 */
	private double[] getRatio(Instances[] instancesPerCluster, int numOfClusters){
		int[] countPerClass;
		double [] clusterRatio = new double[numOfClusters]; // for each cluster: majority/minority
		Instances clusterInstances;
		
		for(int i=0;i<numOfClusters;i++){ //for each cluster
			clusterInstances=instancesPerCluster[i]; 
			countPerClass = clusterInstances.attributeStats(clusterInstances.classIndex()).nominalCounts;
			if (countPerClass[1-this.majority_index]==0){
				clusterRatio[i]=50;
			}
			else{
				clusterRatio[i] = countPerClass[this.majority_index]/countPerClass[1-this.majority_index];
			}
		}
		return clusterRatio;
	}
}

