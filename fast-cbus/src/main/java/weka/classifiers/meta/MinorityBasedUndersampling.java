package weka.classifiers.meta;

import weka.clusterers.Clusterer;
import weka.core.Instances;

public class MinorityBasedUndersampling extends UnderSamplingMethod{

	public MinorityBasedUndersampling(int m) {
		super(m);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public Instances underSample(Instances data, Clusterer clusterer) throws Exception {
		// Add a "cluster" attribute to every instance, according to Clusterer 		
		return super.underSample(data,clusterer);
	}
}
