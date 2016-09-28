
package weka.classifiers.meta;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.PerformanceStats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.clusterers.*;

/**
 * Nir: implement "A novel ensemble method for classifying imbalanced data"
 *  
 * @author Us
 */
public class ClusterBal extends SbcClassifier implements Serializable{
    
    /** whether or not to weight the classification results*/
    private boolean weightedResults;
    /** undersampling size. percentage of majority out of minority. if set to 1 then equal proportion in each cluster */
    private double proportion;
    private Classifier[] classifierForCluster;
    /** Array for storing the generated base classifiers. */
    protected Classifier[] m_Classifiers;
    protected Instances instancesSet;
    /** set weights for instances of each cluster to be used to build corresponding model*/
    private double [] min_prob_minority;
    private double [] max_distance_minority;
    private Instances centroids;
    private int numOfClusters;    
    /* the attribute index of the class**/
    private int classIndex;
    private double[][] maxAttsVals;
    private double [] maxDist;
    /** bound of a cluster is one that leaves this proportion of minority within*/
    private double cutOff;// the minority instance whose distance from centroid is the cutOff's percentage will be considered as most distant in that cluster. to remove anomalies.
    //Map vdmMap;
    private int ClassifiedOutCnt=0;
    int numClassValues;
    private Instances [] minorityTrainingSetsArray;
    protected double[][] m_Ranges;
    EuclideanDistance ed;
    
    // FOr debug
    private int outOfClusterFalse;
    private int outOfClusterCorrect;
    private int cluster1False;
    private int cluster0False;
    private int cluster1True;
    private int cluster0True;
    
    private Map<String, Integer> lineToClass;
    
    
    //private DistanceFunction [] m_DistanceFunctionArray; // one for each cluster
    private boolean printStats;
            
    public ClusterBal() throws Exception{
        super();
        this.underSamplerer = new MinorityBasedUndersampling(1);        
        this.numClassValues=2;
        this.weightedResults=true;
        this.proportion=1;
        this.ed = new EuclideanDistance();
        ed.setDontNormalize(false); // check
        
        SimpleKMeans SKM = new SimpleKMeans();
        SKM.setDistanceFunction(ed);
        this.m_Clusterer= new MakeDensityBasedClusterer(SKM);
        this.cutOff=0.99;  
        
    }
    
    /**
     * set the proportion of majority out of minority
     * @param value
     */
    public void setProportion(double value){
        this.proportion=value;
    }
    public double getProportion(){
        return this.proportion;
    }
    
    /**
     * set whether weight the classification results
     * @param value
     */
    public void setWeightedResult(boolean value){
        this.weightedResults=value;
    }
    public boolean getWeightedResult(){
        return this.weightedResults;
    }
    
    public void setCutOff(double value){
        this.cutOff=value;
    }
    public double getCutOff(){
        return this.cutOff;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {    
        outOfClusterCorrect=0;
        outOfClusterFalse=0;
        cluster1False=0;
        cluster0False=0;
        cluster1True=0;
        cluster0True=0;
        printStats=false;
        instancesSet = new Instances(data, 1);
        // Find the class of the minority. can skip this process if the minority class is given
        minorityClass = findMinorityClass(data); 
        Instances onlyMinority = new Instances(data, data.size());
        Instances onlyMajority = new Instances(data, data.size());                
        splitInstancePerClass(onlyMinority, onlyMajority, data, minorityClass);
        
        Remove remove = new Remove();
        int[] attsArray = new int[1];
        attsArray[0]=onlyMajority.classIndex();
        remove.setAttributeIndicesArray(attsArray);
        remove.setInputFormat(onlyMajority);
        Instances onlyMajorityNoClass = Filter.useFilter(onlyMajority, remove) ;
        
        this.m_Clusterer.buildClusterer( onlyMajorityNoClass );
        buildModelsForClusters(onlyMajority, onlyMinority);

    }
    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
    	printStats=false;
        int expectedClassValue=0;
        if (printStats==true)
           {
		            StringBuffer instanceBuffer = new StringBuffer("");
		            // Search for the same instance
		            for(int i=0;i<instance.numAttributes();i++){
		                if(instance.classIndex()!=i){
		                    double val =instance.value(i);
		                    if(instance.attribute(i).isNumeric()){
		                        int intVal = (int)val;
		                        instanceBuffer.append(intVal);
		                    }
		                    else
		                        instanceBuffer.append(instance.attribute(i).value((int)val));
		                    instanceBuffer.append(", ");                
		                }
		            }
		            String instanceStr = instanceBuffer.toString();
		            
		            expectedClassValue = lineToClass.get(instanceStr);
           }
        
        instancesSet.clear();
        instancesSet.add(instance);
        Remove remove = new Remove();
        int[] attsArray = new int[1];
        attsArray[0]=instancesSet.classIndex();
        remove.setAttributeIndicesArray(attsArray);
        remove.setInputFormat(instancesSet);
        Instances tmpSet = Filter.useFilter(instancesSet, remove);
        Instance toClusterInst = tmpSet.get(0);
        //double [] distancesFromClusters= getDistances(toClusterInst, minorityTrainingSetsArray);
        double [] distancesFromClusters= getDistances_by_ed(toClusterInst, tmpSet, false); // check        
        int clusterIndex=getCloseCluster(distancesFromClusters, false);// check
        //int clusterIndex=m_Clusterer.clusterInstance(toClusterInst);   
        
        /*******    weighting classifiers according to corresponding clusters' distance  *****/        
        
        // the instance is within clusters bounds
        double[] classificationResult;
        if (this.weightedResults==false){
            classificationResult =  m_Classifiers[clusterIndex].distributionForInstance(instance);
        }
        else{
            int numOfModels =m_Clusterer.numberOfClusters();
            
            double prob;    
            
            double [] resultPerModelOnMinority=new double[numOfModels];
            double [] resultPerModelOnMajority=new double[numOfModels];
            
            
            double probSum=0;
            for (int i=0;i<numOfModels;i++){
            	prob= distancesFromClusters[i];
                
                probSum+=(1/prob);
                resultPerModelOnMinority[i]=(1/prob)* this.m_Classifiers[i].distributionForInstance(instance)[this.minorityClass];
                resultPerModelOnMajority[i]=(1/prob)* this.m_Classifiers[i].distributionForInstance(instance)[1-this.minorityClass];
                
                
            }
            double minorityScore=0;
            double majorityScore=0;
            for (int i=0;i<numOfModels;i++){
                minorityScore+=resultPerModelOnMinority[i];
                majorityScore+=resultPerModelOnMajority[i];
            }
            
            // Normalize sum to verify that returned classification probability is between 0 and 1
            minorityScore=minorityScore/probSum;
            majorityScore=majorityScore/probSum;

            classificationResult = new double[numClassValues];
            classificationResult[this.minorityClass]=minorityScore;
            classificationResult[1-this.minorityClass]=majorityScore;
            
        }
        
        if (printStats==true)
        {
		            int selectedClass;
		            if(classificationResult[0]>=0.5)
		                selectedClass = 0;
		            else
		                selectedClass = 1;
		            /*
		            if(clusterIndex==0){
		                if(selectedClass!=expectedClassValue)
		                    cluster0False++;
		                else
		                    cluster0True++;
		            }
		            else{ // Cluster 1
		                if(selectedClass!=expectedClassValue)
		                    cluster1False++;
		                else
		                    cluster1True++;            
		            }
		            
		                    
		            if(outOfClusterCorrect+outOfClusterFalse+cluster0False+cluster0True+cluster1False+cluster1True >= 1931+158+2808+1135){
		                System.out.println("NoCluster("+ outOfClusterCorrect+","+outOfClusterFalse +")"+
		                    "Cluster0("+ cluster0True+","+cluster0False +")"+
		                    "Cluster1("+ cluster1True+","+cluster1False +")");
		            }*/

        }
        return classificationResult;
    }
    
    
    
    
    public String toString()
    {    
        return "to do";
    }
    
    /**
     * build model for each corresponding cluster, keep in arrayOfModels
     * each model will be trained based on the majority samples of the cluster and all minority instances  
     * @param data
     * @param onlyMajority
     * @param onlyMinority
     * @return
     * @throws Exception
     */
    private void buildModelsForClusters(Instances onlyMajority, Instances onlyMinority) throws Exception{
        if (this.m_Classifier == null) {
            throw new Exception("A base classifier has not been specified!");
        }
    	this.classIndex=onlyMajority.classIndex();
        this.centroids= ((SimpleKMeans)((MakeDensityBasedClusterer)this.m_Clusterer).getwrappedClusterer()).getClusterCentroids();
        this.numOfClusters = this.m_Clusterer.numberOfClusters();
        Instances[] trainingSetsArray = new Instances[numOfClusters];// training set for each classifier model
        int[][] countersArray = new int[numOfClusters][2]; // count num of instances at each training model 0=minority 1=majority
        // init each set of instances in the array of instances, by setting header information of data
        for (int i=0;i<numOfClusters;i++){
            trainingSetsArray[i]= new Instances(onlyMajority,onlyMajority.size());
        }
        this.m_Classifiers = AbstractClassifier.makeCopies(this.m_Classifier, numOfClusters);
        /******************MAJORITY**************************/
        // for each majority instance -  set its corresponding cluster.
        double dist;
        initializeRanges(onlyMajority);
        //EuclideanDistance ed = new EuclideanDistance();
        //ed.setInstances(onlyMinority);
        int clusterIndex;
        double [] distancesFromClusters;
        for(Instance inst : onlyMajority){
        	//clusterIndex=this.m_Clusterer.clusterInstance(inst); // check
        	distancesFromClusters= getDistances_by_ed(inst, onlyMajority, false); // check
        	clusterIndex=getCloseCluster(distancesFromClusters, false); // since maxDist is not assigned, norm=false
        	trainingSetsArray[clusterIndex].add(inst); // allocate to the relevant training set
        	countersArray[clusterIndex][0]++;
        	//dist=ed.distance(inst, centroids.get(clusterIndex)); // check
        }             
        
    
        /******************MINORITY**************************/
        initializeRanges(onlyMinority);
        // add each minority instance to all models - for training
        //EuclideanDistance ed = new EuclideanDistance();
        //ed.setInstances(onlyMinority);
        for (int i=0;i<m_Clusterer.numberOfClusters();i++){ 
	        for(Instance inst : onlyMinority){
	        	trainingSetsArray[i].add(inst); // allocate to the relevant training set
	        }
        }
        
	        
	        
	        
        // build the models
        for (int i1=0;i1<this.m_Classifiers.length;i1++){
            this.m_Classifiers[i1].buildClassifier(trainingSetsArray[i1]);
        }
        
    }
    
    private double distance1 (Instance centroid, Instance inst, int clusterIndex){ // instances must not contain the class attribute since the centroid does not.
        double distance = 0;
        Attribute att ;
        for (int i=0; i<inst.numAttributes();i++)
        {
                att=inst.attribute(i);  
		    	double centroidVal = centroid.value(att);
			    double instVal = inst.value(att);
			    if (att.isNumeric()) {
			    	distance += Math.pow(centroidVal - instVal, 2);
			    } else {
				//distance += ((double[][]) vdmMap.get(att.name()))[(int) centroidVal][(int) instVal] ;
			    }
        }
        return distance;
    }
	    
    /*
     * normalized distance
     */
    private double distance2 (Instance centroid, Instance inst,EuclideanDistance ed){ 
    	return ed.distance(inst, centroid);
    }
	    
    
	         	

    // get distance between centroid and instance from the same cluster. norm by artificial minority inst
    private double distance (Instance centroid, Instance inst, int clusterIndex){
        
          double distance = 0;
          Attribute att ;
          for (int i=0; i<inst.numAttributes();i++)
          {
              if(i!=classIndex){
                  att=inst.attribute(i);                  
                  double aVal = centroid.value(att);
                  double bVal = inst.value(att);
                  if (att.isNumeric()) {                        
                        if(maxAttsVals[clusterIndex][i] !=0){
                            distance += Math.pow( (aVal - bVal)/maxAttsVals[clusterIndex][i], 2);
                        }
                        else{
                            distance += 0;
                        }
                  } else { // nominal
                      if (aVal==bVal){
                          distance += 0;
                      }
                      else{
                          distance += 1;
                      }
                  }
              }
          }
          distance=distance/inst.numAttributes();
          distance = Math.pow(distance, .5);
          return distance;
    }
    
    public static int factorial(int n)
    {
        if (n == 0) return 1;
        return n * factorial(n-1);
    }
    
    /*
     * using the clusterer for that will always yield the nearest cluster. however, in classification
     * we wish to know whether instances are outside the cluster - they are set as majority  without employing classifiers
     */
    private int getCloseCluster(double [] distancesFromClusters, boolean should_normalize) throws Exception{
    	int clusterNum =-1;
    	double minProb=Double.MAX_VALUE;
    	double prob;
    	for (int i=0;i<m_Clusterer.numberOfClusters();i++){ 
    		if (should_normalize){
    			prob= getDistanceFromCluster(i, distancesFromClusters[i]); // check - this is norm
    		}else{
    			prob= distancesFromClusters[i];
    		}
    		if (prob<minProb){
    			clusterNum=i;
    			minProb=prob;
    		}
    	}
    	return clusterNum;   	
    }
    
    /*
     * get distances of instance from each cluster 
     */
    private double [] getDistances(Instance currentForClusterer) throws Exception{
    	double [] distancesFromClusters= new double [m_Clusterer.numberOfClusters()];
    	for (int i=0;i<m_Clusterer.numberOfClusters();i++){
    		distancesFromClusters[i] = distance1(centroids.get(i),currentForClusterer,i);
    	}
    	return distancesFromClusters;
    }
    
    
    /*
     * get norm distances of instance from each cluster 
     */
    private double [] getDistances(Instance currentForClusterer, Instances [] minorityTrainingSetsArray) throws Exception{
    	double [] distancesFromClusters= new double [m_Clusterer.numberOfClusters()];
    	for (int i=0;i<m_Clusterer.numberOfClusters();i++){
    		EuclideanDistance ed = new EuclideanDistance();
            ed.setInstances(minorityTrainingSetsArray[i]);            	
    		distancesFromClusters[i] = distance2(currentForClusterer , centroids.get(i),ed);
    	}
    	return distancesFromClusters;
    }
    
    private double [] getDistances_by_ed(Instance currentForClusterer, Instances data, boolean m_DontNormalize ) throws Exception{
    	double [] distancesFromClusters= new double [m_Clusterer.numberOfClusters()];
    	for (int i=0;i<numOfClusters; i++){
    		distancesFromClusters[i] = distance_by_ed(centroids.get(i),currentForClusterer, data,  m_DontNormalize);
    	}
    	return distancesFromClusters;
    }
    
    /*
     * used with majority instances, since the clusterer is trained with minority.
     */
    private boolean isInsideCluster (int clusterIndex, Instance toClusterInst, double distToClosestCluster) { 
    	if ( getDistanceFromCluster(clusterIndex, distToClosestCluster) > 2 ){ // outside cluster.  check the value
    		return false;
    	}else{
    		return true;
    	}
    }

	/*
	 * get distance from given cluster
	 */    
    private double getDistanceFromCluster (int clusterIndex, double distanceFromCluster){
    	return distanceFromCluster / maxDist[clusterIndex];
    }


    
    public double distance_by_ed(Instance first, Instance second, Instances data, boolean m_DontNormalize) {
        double distance = 0;
        int numAttributes1 = data.numAttributes();
        int numAttributes2 = data.numAttributes();
        int classIndex = data.classIndex();
                
        for (int p1 = 0, p2 = 0; p1 < numAttributes1-1 && p2 < numAttributes2-1; ) {
        	if (p1==classIndex){
        		p1++;
        		numAttributes1++;
        	}
        	if (p2==classIndex){
        		p2++;
        		numAttributes2++;
        	}
           
          double diff;
          
    	diff = difference(p1, first.value(p1),second.value(p2), data , m_DontNormalize);
    	p1++;
    	p2++;
          
          distance = updateDistance(distance, diff);
        }
        //return distance;  // check
        return Math.sqrt(distance);
      }

    /**
     * Updates the current distance calculated so far with the new difference
     * between two attributes. The difference between the attributes was 
     * calculated with the difference(int,double,double) method.
     * 
     * @param currDist	the current distance calculated so far
     * @param diff	the difference between two new attributes
     * @return		the update distance
     * @see		#difference(int, double, double)
     */
    protected double updateDistance(double currDist, double diff) {
      double	result;
      
      result  = currDist;
      result += diff * diff ;
      //result += Math.abs(diff); //check
      
      return result;
    }
    
    /**
     * Computes the difference between two given attribute
     * values.
     * 
     * @param index	the attribute index
     * @param val1	the first value
     * @param val2	the second value
     * @return		the difference
     */
    protected double difference(int index, double val1, double val2, Instances data, boolean m_DontNormalize) {
    	int R_MIN=0;
		 int R_MAX=1;
		 int R_WIDTH=2;
    	switch (data.attribute(index).type()) {
        case Attribute.NOMINAL:
          //return ((double[][]) vdmMap.get(data.attribute(index).name()))[(int) val1][(int) val2] ; // check
          if (Utils.isMissingValue(val1) ||
             Utils.isMissingValue(val2) ||
             ((int) val1 != (int) val2)) {
            return 1;
          }
          else {
            return 0;
          }
          
        case Attribute.NUMERIC:
            return (!m_DontNormalize) ? 
                	 (norm(val1, index) - norm(val2, index)) :
                	 (val1 - val2);
        default:
          return 0;
      }
    }
    
    protected double norm(double x, int i) {
    	int R_MIN=0;
		 int R_MAX=1;
		 int R_WIDTH=2;
        if (Double.isNaN(m_Ranges[i][R_MIN]) || (m_Ranges[i][R_MAX] == m_Ranges[i][R_MIN]))
          return 0;
        else
          return (x - m_Ranges[i][R_MIN]) / (m_Ranges[i][R_WIDTH]);
      }
    
    

	/**
	 * Initializes the ranges using all instances of the dataset.
	 * Sets m_Ranges.
	 * 
	 * @return 		the ranges
	 */
	public double[][] initializeRanges(Instances m_Data) {
	  if (m_Data == null) {
	    m_Ranges = null;
	    return m_Ranges;
	  }
	  
	  int numAtt = m_Data.numAttributes();
	  double[][] ranges = new double [numAtt][3];
	  
	    // initialize ranges using the first instance
	    updateRangesFirst(m_Data.instance(0), numAtt, ranges);
	  
	  
	  // update ranges, starting from the second
	  for (int i = 1; i < m_Data.numInstances(); i++)
	    updateRanges(m_Data.instance(i), numAtt, ranges);
	
	  m_Ranges = ranges;
	  
	  return m_Ranges;
	}
	
	/**
	 * Used to initialize the ranges. For this the values of the first
	 * instance is used to save time.
	 * Sets low and high to the values of the first instance and
	 * width to zero.
	 * 
	 * @param instance 	the new instance
	 * @param numAtt 	number of attributes in the model
	 * @param ranges 	low, high and width values for all attributes
	 */
	public void updateRangesFirst(Instance instance, int numAtt, double[][] ranges) {
		   int R_MIN=0;
		   int R_MAX=1;
		   int R_WIDTH=2;
		for (int j = 0; j < numAtt; j++) {
	    if (!instance.isMissing(j)) {
	      ranges[j][R_MIN] = instance.value(j);
	      ranges[j][R_MAX] = instance.value(j);
	      ranges[j][R_WIDTH] = 0.0;
	    }
	    else { // if value was missing
	      ranges[j][R_MIN] = Double.POSITIVE_INFINITY;
	      ranges[j][R_MAX] = -Double.POSITIVE_INFINITY;
	      ranges[j][R_WIDTH] = Double.POSITIVE_INFINITY;
	    }
	  }
	}
	
	  /**
	   * Updates the minimum and maximum and width values for all the attributes
	   * based on a new instance.
	   * 
	   * @param instance 	the new instance
	   * @param numAtt 	number of attributes in the model
	   * @param ranges 	low, high and width values for all attributes
	   */
	  public void updateRanges(Instance instance, int numAtt, double[][] ranges) {
	    // updateRangesFirst must have been called on ranges
	   int R_MIN=0;
	   int R_MAX=1;
	   int R_WIDTH=2;
	   
	   
		  for (int j = 0; j < numAtt; j++) {
	      double value = instance.value(j);
	      if (!instance.isMissing(j)) {
	        if (value < ranges[j][R_MIN]) {
	          ranges[j][R_MIN] = value;
	          ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
	          if (value > ranges[j][R_MAX]) { //if this is the first value that is
	            ranges[j][R_MAX] = value;    //not missing. The,0
	            ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
	          }
	        }
	        else {
	          if (value > ranges[j][R_MAX]) {
	            ranges[j][R_MAX] = value;
	            ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
	          }
	        }
	      }
	    }
	  }
	  
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
		public static void splitInstancePerClass(Instances onlyMinority,Instances onlyMajority,Instances data, int minorityClass){
			for(Instance inst : data){
				if (inst.classValue()==minorityClass){
					onlyMinority.add(inst);
				}else{
					onlyMajority.add(inst);
				}							
			}
		}
}