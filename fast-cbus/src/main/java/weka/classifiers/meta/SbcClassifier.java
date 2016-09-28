package weka.classifiers.meta;

import java.awt.image.SampleModel;
import java.io.Serializable;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;

/**
 * The SBC classifier.
 * First clusters the data.
 * Then undersamples the data based on the samples per cluster.
 * Finaly, gives the resulting dataset to a classifier.
 * 
 * @author Users
 */
public class SbcClassifier extends SingleClassifierEnhancer implements Serializable{
		
	/* The clusterer */
	protected Clusterer m_Clusterer;

	/* Undersampling method */
	protected UnderSamplingMethod underSamplerer;
	
	/* The filter that clusters the data */
	protected AddCluster m_ClusteringFilter;
	
	/* The class of the minority */
	protected int minorityClass;
	
	public SbcClassifier()
	{
		this.m_Classifier = new J48();	// Default classifier
		this.m_Clusterer = new SimpleKMeans(); // Default clusterer
		this.underSamplerer = new UnderSamplingMethod(1); // Default undersamplerer
		this.m_ClusteringFilter = new AddCluster();
		this.m_ClusteringFilter.setClusterer(this.m_Clusterer);
	}
	
	public void buildClassifier(Instances data) throws Exception
	{	
		// Find the class of the minority
		minorityClass = ImbalancedUtils.findMinorityClass(data);
		
		// Runs a clustering algorithm and adds a "cluster" attribute to each instance, with its cluster
		this.m_ClusteringFilter.setInputFormat(data);
		data = Filter.useFilter(data, this.m_ClusteringFilter);

		// under sampling
		Instances undersampledData = underSamplerer.underSample(data,this.m_Clusterer);		

		// Classify with undersampling data
		undersampledData.deleteAttributeAt(undersampledData.attribute("cluster").index());		
		m_Classifier.buildClassifier(undersampledData);
	}
	
	public String toString()
	{	
		return "Sbc["+m_Classifier.toString()+","+m_Clusterer.toString()+"]";
	}
	
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return m_Classifier.distributionForInstance(instance);
	}
	
	
	// ------ Code for GUI property sheet (mostly copied from SingleClustererEnhancer -----
	/**
	 * Returns the tip text for this property
	 * 
	 * @return 		tip text for this property suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public String clustererTipText() {
		return "The base clusterer to be used.";
	}

	/**
	 * Set the base clusterer.
	 *
	 * @param value 	the classifier to use.
	 */
	public void setClusterer(Clusterer value) {
		m_Clusterer = value;
		m_ClusteringFilter.setClusterer(m_Clusterer);
	}

	/**
	 * 	Get the clusterer used as the base clusterer.
	 *
	 * @return 		the base clusterer
	 */
	public Clusterer getClusterer() {
		return m_Clusterer;
	}
  
	/**
	 * 	Gets the clusterer specification string, which contains the class name of
	 * the clusterer and any options to the clusterer
	 *
	 * @return 		the clusterer string
	 */
	protected String getClustererSpec() {
		String	result;
		Clusterer 	clusterer;
		
		clusterer = getClusterer();
		result    = clusterer.getClass().getName();
    
		if (clusterer instanceof OptionHandler)
			result += " " + Utils.joinOptions(((OptionHandler) clusterer).getOptions());
    
		return result;
	}	
}
