package weka.classifiers.meta;

import java.util.Iterator;

import weka.clusterers.Clusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A set of tools for working with imbalanced datasets.
 * @author Nir & Roni
 */
public class ImbalancedUtils {
	/**
	 * Returns the class that is the minority
	 * @param data the set of instances to check
	 * @return the index of the class that is the minority
	 */
	public static int findMinorityClass(Instances data)
	{
		int[] instancesPerClass = new int[data.numClasses()];
		for(Instance sample : data)			
			instancesPerClass[(int) sample.classValue()]++;

		int minInstances=data.size();
		int minClass=-1;
		for(int i=0;i<data.numClasses();i++)
			if(instancesPerClass[i]<minInstances){
				minInstances=instancesPerClass[i];
				minClass=i;
			}
		return minClass;
	}
	
	/**
	 * Remove instance from a given class
	 * @param data, the data set to filter
	 * @param classToRemove, instances of this class are removed
	 */
	public static void filterInstanceOfClass(Instances data, int classToRemove){
		Iterator<Instance> iterator=data.iterator();
		Instance instance;
		while(iterator.hasNext()){
			instance = iterator.next();
			if(instance.classValue()==classToRemove)
				iterator.remove();
		}
	}
	
	/**
	 * 	For every cluster store instanced correspond with
	 * @param data set, with a "cluster" attribute
	 * @return an array of Instances objects, divided by their cluster
	 * @throws Exception
	 */
	public static Instances[] getClustersInstances(Instances data) throws Exception{
		Attribute clusterAttribute = data.attribute("cluster");
		int numOfClusters = clusterAttribute.numValues();
		int clusterIndex;
		Instances[] instancesPerCluster = new Instances[numOfClusters]; // in each index(cluster) count number of instances from train
		
		// Creates a new, empty Instances object for every cluster
		for(int i=0;i<instancesPerCluster.length;i++){
			instancesPerCluster[i] = new Instances(data,0,0);
		}
		// TODO: Need to create an instance of every Instances object in the array
		// TODO: Then, need to remove the class attribute before clustering (this is my current guess)
		for(Instance instance : data){
			clusterIndex = (int)instance.value(clusterAttribute);
			instancesPerCluster[clusterIndex].add(instance);			
		}
		return instancesPerCluster;
	}	
}
