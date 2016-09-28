package weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import org.w3c.dom.Attr;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.classifiers.functions.LinearRegression;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.OptionHandler;
import weka.core.Summarizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

import java.io.Serializable;
/**
 * Implementation of a regression tree. 
 */

public class RegressionTree extends SingleClassifierEnhancer implements Serializable{
    /** The number of instances for continuing to split the dataset. */
    protected int m_minInstancesForSplit = 10;
    
    /** for serialization */
    static final long serialVersionUID = 1262712745870398755L;
    
    /** The regression tree model */
    protected RegressionTreeNode m_ModelRoot;
    
    /** splitting subsets for numeric attributes*/
    protected int splitSubsets = 2;
    
    /** count number of leaves on numeric*/
    protected int cnt_leafModelNum = 0;
    
    /** count splitting nodes*/
    protected int cnt_splitNodes = 0;
    
    public RegressionTree(){
    	m_Classifier = new LinearRegression(); 	
    }
    
    /** Getter/Setter Functions to handle options **/
    public int getMinInstancesForSplit(){
    	return m_minInstancesForSplit;
    }
    public void setMinInstancesForSplit(int minInstancesForSplit){
    	m_minInstancesForSplit=minInstancesForSplit;
    }
    
    /**
     * Build the regression tree
     */
	public void buildClassifier(Instances data) throws Exception{
	    // can classifier handle the data?
	    getCapabilities().testWithFail(data);
		
	    // remove instances with missing class
	    Instances filteredData = new Instances(data);
	    filteredData.deleteWithMissingClass();

	    // Build tree recursively   
	    Vector<Attribute> attributes = new Vector<Attribute>(Collections.list(data.enumerateAttributes()));
	    m_ModelRoot = buildTree(data, attributes);
	}
	
	/**
	 * Build the tree for the given data set, return the root node of that tree.
	 * @param data the training set
	 * @param attributes the attributes from which to select 
	 * @return the root of the learned regression tree.
	 */
	private RegressionTreeNode buildTree(Instances data,Vector<Attribute> attributes) throws Exception
	{
		// generate leaf node
		if(data.size()<m_minInstancesForSplit || attributes.size()==0){
			return buildLeafNode(data);
		}
			
	
		// Else build a tree node - call this function recursively:
		// Select the split attribute
		Attribute curAtt;
		Attribute selectedAttribute=null; // index of the selected attribute
		double winningsplitPoint=-1; // index of best attribute's value for splitting (if numeric)
		Instances [] WinningInstancesSubsetsArray=null; // splitting the node, according to winning attribute
		
		double lowestMSE=Integer.MAX_VALUE; // MSE of the selected attribute
		double currMSE; // current attribute's MSE
		Instances [] instancesSubsetsArray; // splitting the node
		for (int attIndex=0;attIndex<attributes.size();attIndex++)
		{
			curAtt=attributes.get(attIndex);
			// numeric attribute
			if (curAtt.isNumeric()) 
			{
				// for each current attribute's value calculate joint MSE
				for(int i=0;i<data.size();i++)
				{
					double splitPoint = data.get(i).value(curAtt);
					instancesSubsetsArray=split(data,curAtt,splitPoint);
					currMSE=calculatedJointMSE(instancesSubsetsArray);
					if (currMSE<lowestMSE){
						selectedAttribute=curAtt;
						winningsplitPoint=splitPoint;
						WinningInstancesSubsetsArray=instancesSubsetsArray;
						lowestMSE=currMSE;
					}
				}
			}
			else // non-numeric attribute (nominal/string)
			{
				// each attribute's value is a split point 
				instancesSubsetsArray=split(data,curAtt,-1);
				currMSE=calculatedJointMSE(instancesSubsetsArray);
				if (currMSE<lowestMSE){
					selectedAttribute=curAtt;
					WinningInstancesSubsetsArray=instancesSubsetsArray;
					lowestMSE=currMSE;
				}
			}
		}
		// Create a vector of all the attributes except the selected attribute. 
		// The children of this node will select splitting attributes from this subset of attributes. 
		Vector<Attribute> subAttributes = new Vector<Attribute>(attributes);
		subAttributes.remove(selectedAttribute);
		
		// Build a tree node for every subset of training set, split according to the selected attribute
		Vector<RegressionTreeNode> children = new Vector<RegressionTreeNode>();		
		for(Instances subData : WinningInstancesSubsetsArray){
			if (subData.size()==0)// learn regression model from all father-splitNode instances
				children.add(buildLeafNode(data));
			else
				children.add(buildTree(subData, subAttributes));
		}
		cnt_splitNodes++;
		return new RegressionTreeSplitNode(winningsplitPoint,children,selectedAttribute);
	}
		
	/**
	 * Build a leaf node from the given set of instances.
	 * @param data: the training set
	 * @return the learned leaf node
	 */
	private RegressionTreeLeaf buildLeafNode(Instances data) throws Exception
	{
		Classifier newClassifier = AbstractClassifier.makeCopy(m_Classifier);
		cnt_leafModelNum++;
		newClassifier.buildClassifier(data);
		return new RegressionTreeLeaf(newClassifier,cnt_leafModelNum);
	}
	
	/**
	 * Classify a single instance by going down the regression tree.
	 */
	public double classifyInstance(Instance instance)
	{
		try{
			return m_ModelRoot.classifyInstance(instance);
		}
		catch(Exception exception){
			throw new RuntimeException("Problem in classifying");
		}
	}

    /**
     * Returns a description of the regression model tree (tree structure and regression models at leaves)
     * @return describing string
     */
	public String toString(){
		// Print out the regression tree.
		try{
		    StringBuffer text = new StringBuffer();
		    StringBuffer leaves_text = new StringBuffer();
		    
		    if (m_ModelRoot.IsLeaf()) {
			text.append(": ");
			text.append("\n\nRM_"+((RegressionTreeLeaf)m_ModelRoot).getModelNumber()+":"+((RegressionTreeLeaf)m_ModelRoot).m_Classifier.toString());
		    } else {
			dumpTree(0,text,(RegressionTreeSplitNode)m_ModelRoot,leaves_text);	    	    
		    }
		    text.append("\n\nNumber of Leaves : \t"+Integer.toString(cnt_leafModelNum)+"\n");
		    text.append("\nSize of the Tree : \t"+Integer.toString(cnt_splitNodes)+"\n");
		    text.append(leaves_text.toString());
		        
		    //This prints regression models after the tree, comment out if only tree should be printed
		    //text.append(modelsToString());
		    return text.toString();
		} catch (Exception e){
		    return "Can't print logistic model tree";
		}
		
	        
	    }

    /**
     * Help method for printing tree structure.
     *
     * @throws Exception if something goes wrong
     */
    protected void dumpTree(int depth,StringBuffer text,RegressionTreeSplitNode node,StringBuffer leaves_text) 
	throws Exception {
	
	for (int i = 0; i < node.children.size(); i++) {
	    text.append("\n");
	    for (int j = 0; j < depth; j++)
	    	text.append("|   ");
	    text.append(node.splitAttribute.name());
	    text.append(rightSide(i, node.splitAttribute, node.SplittingValue));
	    if ( node.children.get(i).IsLeaf()){
		text.append(": ");
		text.append("RM_"+((RegressionTreeLeaf)node.children.get(i)).getModelNumber());
	    leaves_text.append("\n\nRM_"+((RegressionTreeLeaf)node.children.get(i)).getModelNumber()+":"+((RegressionTreeLeaf)node.children.get(i)).m_Classifier.toString());
	    }else
	    	dumpTree(depth+1,text, (RegressionTreeSplitNode)node.children.get(i),leaves_text);
	}
    }
	
    
    /**
     * Prints the condition satisfied by instances in a subset.
     *
     * @param index of subset 
     * @param attribute
     */
    public final String rightSide(int index,Attribute attribute,double m_splitPoint) {

      StringBuffer text;

      text = new StringBuffer();
      if (attribute.isNominal())
        text.append(" = "+
        	attribute.value(index));
      else
        if (index == 0)
  	text.append(" <= "+
  		    Utils.doubleToString(m_splitPoint,6));
        else
  	text.append(" > "+
  		    Utils.doubleToString(m_splitPoint,6));
      return text.toString();
    }

	
	/**
	 * Calculate joint MSE as weighted average
	 * @param data Array of subset instances
	 * @return The joint mean square error for all subsets
	 * @throws Exception
	 */
	
	public double calculatedJointMSE(Instances [] dataArray) throws Exception{
		double JointMSE=0;
		double curMSE;
		int cntSubsetInstances;
		int cntInstance=0;
		for (int subsetIndex=0;subsetIndex<dataArray.length;subsetIndex++)
		{
			cntSubsetInstances=dataArray[subsetIndex].numInstances();
			curMSE=calculatedMeanSquareError(dataArray[subsetIndex]);
			JointMSE+=cntSubsetInstances*curMSE;
			cntInstance+=cntSubsetInstances;
		}
		JointMSE=JointMSE/cntInstance; //weighted average
		
		return JointMSE;
	}
	
	/**
	 * A function that calculates the mean square error for a set of instances.
	 * First the base classifier (a regression of some sort) is built.
	 * Then its mean square error is calculated and returned.
	 * @param data The set of data to be inputed to the regression model.
	 * @return The mean square error or running the base regression algorithm on the given dataset
	 * @throws Exception
	 */
	public double calculatedMeanSquareError(Instances data) throws Exception{
		double predictedValue;
		double observedValue;
		double meanSquareError=0;
		if (data.numInstances()==0){
			return 0;
		}
		this.m_Classifier.buildClassifier(data);
		for(Instance instance : data){
			predictedValue = this.m_Classifier.classifyInstance(instance);
			observedValue = instance.classValue();
			meanSquareError = meanSquareError+(predictedValue-observedValue)*(predictedValue-observedValue);
		}
		return meanSquareError/data.numInstances();  
	}

	  /**
	   * Splits the given set of instances into subsets.
	   *
	   * @exception throws Exception
	   */
	  public final Instances [] split(Instances data,Attribute m_att, double m_splitPoint) throws Exception { 

		int m_numSubsets;
		if (m_att.isNominal())
			m_numSubsets=m_att.numValues();
		else
			m_numSubsets=splitSubsets;
		
		Instances [] instances = new Instances [m_numSubsets];
	    Instance instance;
	    int subset, i, j;
	    for (j=0;j<m_numSubsets;j++)
	      instances[j] = new Instances((Instances)data,
						    data.numInstances());
	    for (i = 0; i < data.numInstances(); i++) {
	      instance = ((Instances) data).instance(i);
	      subset = whichSubset(instance, m_att, m_splitPoint);
	      if (subset > -1)
	    	  instances[subset].add(instance);
	    }	    
	    return instances;
	  }


/**
 * Returns index of subset instance is assigned to.
 * Returns -1 if instance is assigned to more than one subset.
 *
 * @exception Exception if something goes wrong
 */
public final int whichSubset(Instance instance,Attribute m_att, double m_splitPoint) 
     throws Exception {

  if (instance.isMissing(m_att))
    return -1;
  else{
    if (m_att.isNominal())
	return (int)instance.value(m_att);
    else
	if (Utils.smOrEq(instance.value(m_att),m_splitPoint))
	  return 0;
	else
	  return 1;
  }
}

/**
 * Represents the nodes of the regression tree.
 */
abstract class RegressionTreeNode implements Serializable
{
	abstract double classifyInstance(Instance instance) throws Exception;
	abstract boolean IsLeaf();
}

/**
 * Non-leaf node.
 */
class RegressionTreeSplitNode extends RegressionTreeNode implements Serializable{
	static final long serialVersionUID = 1812376745870398355L;
	private double SplittingValue; // Splitting value for numeric  (ignore if non-numeric)
	private List<RegressionTreeNode> children; // List of child nodes
	private Attribute splitAttribute; // splitting attribute
	
	public RegressionTreeSplitNode(double SplittingValue, 
			List<RegressionTreeNode> pChildren,
			Attribute pSplitAttribute)
	{
		this.SplittingValue = SplittingValue;
		this.children = pChildren;
		this.splitAttribute=pSplitAttribute;
	}
	
	public double classifyInstance(Instance instance) throws Exception
	{
		int childIndex=whichSubset(instance,splitAttribute,SplittingValue);
		RegressionTreeNode correctChild=children.get(childIndex);
		// Call classifyInstance of that child
		return correctChild.classifyInstance(instance);
	}
	boolean IsLeaf(){
		return false;
	}
}

/**
 * Leaf node.
 */
class RegressionTreeLeaf extends RegressionTreeNode implements Serializable{
	static final long serialVersionUID = 1212376745870398755L;
	private int m_leafModelNum;
	private Classifier m_Classifier;
	public RegressionTreeLeaf(Classifier pClassifier,int pLeafModelNum)
	{
		this.m_Classifier = pClassifier;
		this.m_leafModelNum=pLeafModelNum;
	}
	
	public double classifyInstance(Instance instance) throws Exception
	{
		return m_Classifier.classifyInstance(instance);
	}
	public boolean IsLeaf(){
		return true;
	}
	public String getModelNumber(){
		return Integer.toString(m_leafModelNum); 
	}
}

}





