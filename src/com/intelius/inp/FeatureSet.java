package com.intelius.inp;


import java.util.*;
import iitb.CRF.DataSequence;
import iitb.CRF.Feature;
import iitb.Model.FeatureTypes;
import iitb.CRF.FeatureGeneratorNested;

public class FeatureSet implements FeatureGeneratorNested {

	private static final long serialVersionUID = 1L;
	List<FeatureTypes> features = new ArrayList<FeatureTypes>();
	
	@Override
	public int numFeatures() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void startScanFeaturesAt(DataSequence data, int pos) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Feature next() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String featureName(int featureIndex) {
		// TODO Auto-generated method stub
		return null;
	}

	// the maximum number of sequences grouped together
	@Override
	public int maxMemory() {
		return 1;
	}

	@Override
	public void startScanFeaturesAt(DataSequence data, int prevPos, int pos) {
		// TODO Auto-generated method stub
		
	}

}
