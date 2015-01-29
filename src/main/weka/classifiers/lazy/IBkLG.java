/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    IBkLG.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *    Copyright (C) 2015 Shesha Sreenivasamurthy, University of California, Santa Cruz, USA
 *
 */

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.lang.Math;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

import weka.classifiers.lazy.IBk; /** Super Class */

/**
 * K-nearest neighbors classifier. Can select appropriate value of K based on cross-validation. Can also do distance weighting.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * D. Aha, D. Kibler (1991). Instance-based learning algorithms. Machine Learning. 6:37-66.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Aha1991,
 *    author = {D. Aha and D. Kibler},
 *    journal = {Machine Learning},
 *    pages = {37-66},
 *    title = {Instance-based learning algorithms},
 *    volume = {6},
 *    year = {1991}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -L
 *  Weight neighbors by the log of their distance
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -G
 *  Weight neighbors by a gaussian around them
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -S &lt;Gaussian Standard Deviation&gt;
 *  Standard deviation to be used for the gaussian
 *  (Default = 1)</pre>
 *  
 * <pre> -K &lt;number of neighbors&gt;
 *  Number of nearest neighbors (k) used in classification.
 *  (Default = 1)</pre>
 * 
 * <pre> -E
 *  Minimize mean squared error rather than mean absolute
 *  error when using -X option with numeric prediction.</pre>
 * 
 * <pre> -W &lt;window size&gt;
 *  Maximum number of training instances maintained.
 *  Training instances are dropped FIFO. (Default = no window)</pre>
 * 
 * <pre> -X
 *  Select the number of nearest neighbors between 1
 *  and the k value specified using hold-one-out evaluation
 *  on the training data (use when k &gt; 1)</pre>
 * 
 * <pre> -A
 *  The nearest neighbor search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
 * </pre>
 * 
 <!-- options-end -->
 *
 * @author Stuart Inglis (singlis@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Shesha Sreenivasamurthy (ssreeniv@ucsc.edu)
 * @version $Revision: 10141 $
 */
public class IBkLG
  extends IBk 
  implements OptionHandler {

  /** for serialization. */
  private static final long serialVersionUID = 2998909624652223405L;

  /** Whether the neighbors should be distance-weighted. */
  protected int m_DistanceWeightingLG = WEIGHT_LOG;
  
  /** Standard deviation to be used in a Gaussian weight */
  protected double m_SD;

  /** weight by natural log **/
  public static final int WEIGHT_LOG = 8;
  /** weight by Gaussian Distribution **/
  public static final int WEIGHT_GAUSSIAN = 16;
  
  /** possible instance weighting methods. */
  public static final Tag [] TAGS_WEIGHTING_LG = {
    new Tag(WEIGHT_LOG, "Weight by log(distance)"),
    new Tag(WEIGHT_GAUSSIAN, "Weight by gaussian(distance)")
  };

  /**
   * IBk classifier. Simple instance-based learner that uses the class
   * of the nearest k training instances for the class of the test
   * instances.
   *
   * @param k the number of nearest neighbors to use for prediction
   */
  public IBkLG(int k) {
    init();
    super.setKNN(k);
  }  

  /**
   * IB1 classifer. Instance-based learner. Predicts the class of the
   * single nearest training instance for each test instance.
   */
  public IBkLG() {
    init();
  }

  /**
   * Initialize scheme variables.
   */
  protected void init() {
	super.init();
    setSD(1.0);
    m_DistanceWeightingLG = WEIGHT_LOG;
  }

  /**
   * Gets the distance weighting method used. Will be one of
   * WEIGHT_LOG or WEIGHT_GAUSSIAN
   *
   * @return the distance weighting method used.
   */
  public SelectedTag getDistanceWeighting() {

    return new SelectedTag(m_DistanceWeightingLG, TAGS_WEIGHTING_LG);
  }
  
  /**
   * Sets the distance weighting method used. Values other than
   * WEIGHT_LOG or WEIGHT_GAUSSIAN will be ignored.
   *
   * @param newMethod the distance weighting method to use
   */
  public void setDistanceWeighting(SelectedTag newMethod) {
    
    if (newMethod.getTags() == TAGS_WEIGHTING_LG) {
      m_DistanceWeightingLG = newMethod.getSelectedTag().getID();
    }
  }
  
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String SDTipText() {
    return "Standard Deviation to be used by Gaussian with zero mean";
  }
  
  /**
   * Set the standard deviation for learner to use.
   *
   * @param sd standard deviation of the gaussian.
   */
  public void setSD(double sd) {
    m_SD = sd;
  }


  /**
   * Gets the standard deviation used by learner.
   *
   * @return standard deviation of the gaussian.
   */
  public double getSD() {
    return m_SD;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(7);

    newVector.addElement(new Option(
    	  "\tWeighted Neighbors by log (distance) ",
    	  "L", 0, "-L"));
    newVector.addElement(new Option(
      	  "\tWeighted Neighbors by gaussian (distance) ",
      	  "G", 0, "-G"));
    newVector.addElement(new Option(
          "\tStandard Deviation for gaussian.(Default = 1.0)\n",
    	  "S", 1,"-S <sd>"));
    newVector.addElement(new Option(
	      "\tNumber of nearest neighbors (k) used in classification.\n"+
	      "\t(Default = 1)",
	      "K", 1,"-K <number of neighbors>"));
    newVector.addElement(new Option(
          "\tMinimise mean squared error rather than mean absolute\n"+
	      "\terror when using -X option with numeric prediction.",
	      "E", 0,"-E"));
    newVector.addElement(new Option(
          "\tMaximum number of training instances maintained.\n"+
	      "\tTraining instances are dropped FIFO. (Default = no window)",
	      "W", 1,"-W <window size>"));
    newVector.addElement(new Option(
	      "\tSelect the number of nearest neighbors between 1\n"+
	      "\tand the k value specified using hold-one-out evaluation\n"+
	      "\ton the training data (use when k > 1)",
	      "X", 0,"-X"));
    newVector.addElement(new Option(
	      "\tThe nearest neighbor search algorithm to use "+
          "(default: weka.core.neighboursearch.LinearNNSearch).\n",
	      "A", 0, "-A"));

    newVector.addAll(Collections.list(super.listOptions()));
    
    return newVector.elements();
  }

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -L
   *  Weight neighbors by the log of their distance
   *  (use when k &gt; 1)</pre>
   * 
   * <pre> -G
   *  Weight neighbors by Gaussian
   *  (use when k &gt; 1)</pre>
   * 
   * <pre> -K &lt;number of neighbors&gt;
   *  Number of nearest neighbors (k) used in classification.
   *  (Default = 1)</pre>
   * 
   * <pre> -E
   *  Minimize mean squared error rather than mean absolute
   *  error when using -X option with numeric prediction.</pre>
   * 
   * <pre> -W &lt;window size&gt;
   *  Maximum number of training instances maintained.
   *  Training instances are dropped FIFO. (Default = no window)</pre>
   * 
   * <pre> -X
   *  Select the number of nearest neighbors between 1
   *  and the k value specified using hold-one-out evaluation
   *  on the training data (use when k &gt; 1)</pre>
   * 
   * <pre> -A
   *  The nearest neighbor search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String knnString = Utils.getOption('K', options);
    if (knnString.length() != 0) {
      setKNN(Integer.parseInt(knnString));
    } else {
      setKNN(1);
    }
    String windowString = Utils.getOption('W', options);
    if (windowString.length() != 0) {
      setWindowSize(Integer.parseInt(windowString));
    } else {
      setWindowSize(0);
    }
    if (Utils.getFlag('L', options)) {
        setDistanceWeighting(new SelectedTag(WEIGHT_LOG, TAGS_WEIGHTING_LG));
    } else if (Utils.getFlag('G', options)) {
        setDistanceWeighting(new SelectedTag(WEIGHT_GAUSSIAN, TAGS_WEIGHTING_LG));
        String sd = Utils.getOption('S', options);
        setSD(Float.parseFloat(sd));
    } else {
    	setDistanceWeighting(new SelectedTag(WEIGHT_LOG, TAGS_WEIGHTING_LG));
    }
    setCrossValidate(Utils.getFlag('X', options));
    setMeanSquared(Utils.getFlag('E', options));

    String nnSearchClass = Utils.getOption('A', options);
    if(nnSearchClass.length() != 0) {
      String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
      if(nnSearchClassSpec.length == 0) { 
        throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                            "specification string."); 
      }
      String className = nnSearchClassSpec[0];
      nnSearchClassSpec[0] = "";

      setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                  Utils.forName( NearestNeighbourSearch.class, 
                                 className, 
                                 nnSearchClassSpec)
                                        );
    }
    else 
      this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());
    
    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of IBk.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    Vector<String> options = new Vector<String>();
    options.add("-K"); options.add("" + getKNN());
    options.add("-W"); options.add("" + m_WindowSize);
    options.add("-S"); options.add("" + m_SD);
    if (getCrossValidate()) {
        options.add("-X");
    }
    if (getMeanSquared()) {
        options.add("-E");
    }
    if (m_DistanceWeightingLG == WEIGHT_LOG) {
        options.add("-L");
    } else if (m_DistanceWeightingLG == WEIGHT_GAUSSIAN) {
        options.add("-G");
    }

    options.add("-A");
    options.add(m_NNSearch.getClass().getName()+" "+Utils.joinOptions(m_NNSearch.getOptions())); 
    
    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
  }
  
  /**
   * Returns a description of this classifier.
   *
   * @return a description of this classifier as a string.
   */
  public String toString() {

    if (m_Train == null) {
      return "IBk: No model built yet.";
    }
    
    if (m_Train.numInstances() == 0) {
      return "Warning: no training instances - ZeroR model used.";
    }

    if (!m_kNNValid && m_CrossValidate) {
      crossValidate();
    }
    
    String result = "IB1 instance-based classifier\n" +
      "using " + m_kNN;

    switch (m_DistanceWeightingLG) {
    case WEIGHT_LOG:
        result += " log-distance-weighted";
        break;
    case WEIGHT_GAUSSIAN:
    	result += " gaussian-distance-weighted (Mean:0, SD:";
    	result += m_SD;
    	result += ")";
    	break;
    }
    result += " nearest neighbor(s) for classification\n";

    if (m_WindowSize != 0) {
      result += "using a maximum of " 
	+ m_WindowSize + " (windowed) training instances\n";
    }
    return result;
  }

  /*
   * Gaussian distribution.
   */
  protected static double gaussian(double mean, double sd, double x)
	throws Exception {
      return Math.exp(-((x-mean)*(x-mean))/(2*sd*sd)) /
              Math.sqrt(2*Math.PI*sd*sd);
  }
  
  /**
   * Turn the list of nearest neighbors into a probability distribution.
   *
   * @param neighbors the list of nearest neighboring instances
   * @param distances the distances of the neighbors
   * @return the probability distribution
   * @throws Exception if computation goes wrong or has no class attribute
   */
  protected double [] makeDistribution(Instances neighbours, double[] distances)
    throws Exception {

    double total = 0, weight;
    double [] distribution = new double [m_NumClasses];
    
    // Set up a correction to the estimator
    if (m_ClassType == Attribute.NOMINAL) {
      for(int i = 0; i < m_NumClasses; i++) {
	distribution[i] = 1.0 / Math.max(1,m_Train.numInstances());
      }
      total = (double)m_NumClasses / Math.max(1,m_Train.numInstances());
    }

    for(int i=0; i < neighbours.numInstances(); i++) {
      // Collect class counts
      Instance current = neighbours.instance(i);
      distances[i] = distances[i]*distances[i];
      distances[i] = Math.sqrt(distances[i]/m_NumAttributesUsed);
      
      switch (m_DistanceWeightingLG) {
      	case WEIGHT_LOG:
      		weight = -Math.log(distances[i] + 0.0000000001); // Avoid infinity
      		break;
        case WEIGHT_GAUSSIAN:
            weight = gaussian(0.0, m_SD, distances[i]);
            break;
        default:  // WEIGHT_LOG:
        	weight = -Math.log(0.0000000001);
        	break;
      }
      weight *= current.weight();
      try {
        switch (m_ClassType) {
          case Attribute.NOMINAL:
            distribution[(int)current.classValue()] += weight;
            break;
          case Attribute.NUMERIC:
            distribution[0] += current.classValue() * weight;
            break;
        }
      } catch (Exception ex) {
        throw new Error("Data has no class attribute!");
      }
      total += weight;      
    }

    // Normalize distribution
    if (total > 0) {
      Utils.normalize(distribution, total);
    }
    return distribution;
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv should contain command line options (see setOptions)
   */
  public static void main(String [] argv) {
    runClassifier(new IBkLG(), argv);
  }
}
