package org.clustering4ever.clusteringanalysis
/**
 * @author Beck Gaël
 */
import scala.collection.{mutable, immutable}
import scala.math.{sqrt, abs}
import org.clustering4ever.clustering.ClusteringSharedTypes
import org.clustering4ever.types.VectorizationIDTypes._
import org.clustering4ever.types.ClusteringNumberType._
import org.clustering4ever.types.ClusterIDType._
import org.clustering4ever.util.SumVectors
/**
 *
 */
trait ClusteringBinaryStatsAncestor extends ClusteringSharedTypes {
	/**
	 * The number of clustering which was applied 
	 */
	val clusteringNumber: ClusteringNumber
	/**
	 * Number of occurences for each features
	 */
	val occurencesPerFeature: Seq[Int]
	/**
	 * Frequencies for each features
	 */
	val frequencyPerFeature: Seq[Double]
	/**
	 * Number of occurences for each features by ClusterID
	 */
	val occurencesPerFeatureByClusterID: immutable.HashMap[ClusterID, Seq[Int]]
	/**
	 * Frequencies for each features by ClusterID
	 */
	val frequencyPerFeatureByClusterID: immutable.HashMap[ClusterID, Seq[Double]]
	/**
	 *
	 */
	val clustersNumbers = occurencesPerFeatureByClusterID.size
	/**
	 *
	 */
	val featuresNumbers = occurencesPerFeature.size
	/**
	 * HashMap of each feature occurence ratio by clusterID, compared to global features 
	 */
	lazy val occurencesRatioPerFeatureByClusterID: immutable.HashMap[ClusterID, Seq[Double]] = {
		occurencesPerFeatureByClusterID.map{ case (clusterID, featsOcc) => (clusterID, featsOcc.zip(occurencesPerFeature).map{ case (a, b) => a.toDouble / b }) }
	}
	/**
	 *
	 */
	lazy val stdFrequencies: Seq[Double] = {
		import org.clustering4ever.util.VectorsAddOperationsImplicits._
		SumVectors.sumColumnMatrix(
			frequencyPerFeatureByClusterID.map(_._2.zip(frequencyPerFeature).map{ case (v, mean) => 
				val toPow = v - mean
				toPow * toPow
			  }).toSeq
		).map( v => sqrt(v / (clustersNumbers - 1)) )
	}
	/**
	 *
	 */
	def getFeaturesFarFromStd(stdGap: Double = 1D): immutable.HashMap[ClusterID, Seq[Int]] = {
		frequencyPerFeatureByClusterID.map{ case (clusterID, freqs) =>
			(
				clusterID,
				stdFrequencies.zipWithIndex.zip(frequencyPerFeature).zip(freqs).collect{ case (((std, featID), mean), v) if abs(v - mean) > (std * stdGap) => featID }
			)
		}
	}
	/**
	 *
	 */
	lazy val distantFeaturesFromStdByCluster: immutable.HashMap[ClusterID, Seq[Int]] = getFeaturesFarFromStd(1D)
	/**
	 *
	 */
	lazy val farestClusterFromFeaturesDistribution: ClusterID = {
		frequencyPerFeatureByClusterID.maxBy{ case (clusterID, freqs) =>
				frequencyPerFeature.zip(freqs).map{ case (mean, v) => abs(v - mean) }.sum
		}._1
	} 
	/**
	 *
	 */
	lazy val closestClusterFromFeaturesDistribution: ClusterID = {
		frequencyPerFeatureByClusterID.minBy{ case (clusterID, freqs) =>
				frequencyPerFeature.zip(freqs).map{ case (mean, v) => abs(v - mean) }.sum
		}._1
	} 
}
/**
 * This class gather various informations about a specific clustering on binary data 
 */
case class ClusteringBinaryAnalysis(
	val clusteringNumber: ClusteringNumber,
	val occurencesPerFeature: Seq[Int],
	val frequencyPerFeature: Seq[Double],
	val occurencesPerFeatureByClusterID: immutable.HashMap[ClusterID, Seq[Int]],
	val frequencyPerFeatureByClusterID: immutable.HashMap[ClusterID, Seq[Double]]
) extends ClusteringBinaryStatsAncestor
/**
 *
 */
trait EveryClusteringBinaryStatsAncestor extends ClusteringSharedTypes {
	/**
	 * ClusteringBinaryAnalysis by clusteringNumber by vectorizationID
	 */
	val clusteringBinaryAnalysisByClusteringNumberByVectorization: mutable.HashMap[VectorizationID, mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis]]
	/**
	 *
	 */
	def getCBAsOneVecto(vectorizationID: VectorizationID): Option[mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis]] = clusteringBinaryAnalysisByClusteringNumberByVectorization.get(vectorizationID)	
	/**
	 *
	 */
	def addCBA(vectorizationID: VectorizationID, clusteringNumber: ClusteringNumber, cba: ClusteringBinaryAnalysis): Unit = {
		if(clusteringBinaryAnalysisByClusteringNumberByVectorization.contains(vectorizationID)) {
			clusteringBinaryAnalysisByClusteringNumberByVectorization(vectorizationID) += (clusteringNumber -> cba)
		}
		else {
			clusteringBinaryAnalysisByClusteringNumberByVectorization += (vectorizationID -> mutable.HashMap(clusteringNumber -> cba))
		}
	}
	/**
	 *
	 */
	def addCBAs(vectorizationID: VectorizationID, cbas: mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis]): Unit = {
		if(clusteringBinaryAnalysisByClusteringNumberByVectorization.contains(vectorizationID)) {
			clusteringBinaryAnalysisByClusteringNumberByVectorization(vectorizationID) ++= cbas
		}
		else {
			clusteringBinaryAnalysisByClusteringNumberByVectorization += (vectorizationID -> cbas)
		}
	}
	/**
	 * @return an Option of the ClusteringBinaryAnalysis corresponding to given ClusteringNumber
	 */
	def getCBAByClusteringNumber(clusteringNumber: ClusteringNumber): Option[ClusteringBinaryAnalysis] = {
		clusteringBinaryAnalysisByClusteringNumberByVectorization.find(_._2.contains(clusteringNumber)).map(_._2(clusteringNumber))
	}
	/**
	 *
	 */
	def getClusteringMaximazingAverageFeaturesStd(vectorizationID: VectorizationID): ClusteringBinaryAnalysis = {
		val cbas: mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis] = getCBAsOneVecto(vectorizationID).get
		cbas.maxBy{ case (clusteringNumber, cba) =>
			cba.stdFrequencies.sum / cba.featuresNumbers
		}._2
	}
}
/**
 *
 */
case class EveryClusteringBinaryAnalysis(
	val clusteringBinaryAnalysisByClusteringNumberByVectorization: mutable.HashMap[VectorizationID, mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis]] = mutable.HashMap.empty[VectorizationID, mutable.HashMap[ClusteringNumber, ClusteringBinaryAnalysis]]
) extends EveryClusteringBinaryStatsAncestor