package clustering4ever.spark.clustering.kmodes

import scala.collection.{immutable, mutable}
import scala.util.Random
import scala.annotation.meta.param
import scala.reflect.ClassTag
import scala.math.{min, max}
import org.apache.spark.{SparkContext, HashPartitioner}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import clustering4ever.math.distances.BinaryDistance
import clustering4ever.math.distances.binary.Hamming
import clustering4ever.scala.clusterizables.BinaryClusterizable
import clustering4ever.spark.clustering.KCommonsSparkVectors
import scala.language.higherKinds
/**
 * @author Beck Gaël
 * The famous K-Means using a user-defined dissmilarity measure.
 * @param data : an Seq with and ID and the vector
 * @param k : number of clusters
 * @param epsilon : minimal threshold under which we consider a centroid has converged
 * @param iterMax : maximal number of iteration
 * @param metric : a defined dissimilarity measure, it can be custom by overriding BinaryDistance distance function
 */
class KModes[
	ID: Numeric,
	O,
	V[Int] <: Seq[Int],
	Cz[ID, O, V <: Seq[Int]] <: BinaryClusterizable[ID, O, V, Cz[ID, O, V]],
	D <: Hamming[V[Int]]
](
	@transient val sc: SparkContext,
	data: RDD[Cz[ID, O, V[Int]]],
	k: Int,
	epsilon: Double,
	maxIter: Int,
	metric: D = new Hamming[V[Int]],
	initializedCenters: mutable.HashMap[Int, V[Int]] = mutable.HashMap.empty[Int, V[Int]],
	persistanceLVL: StorageLevel = StorageLevel.MEMORY_ONLY
)(implicit ct: ClassTag[Cz[ID, O, V[Int]]], ct2: ClassTag[V[Int]]) extends KCommonsSparkVectors[ID, Int, V, Cz[ID, O, V[Int]], D](data, metric, k, initializedCenters, persistanceLVL) {
	def run(): KModesModel[ID, O, V[Int], Cz[ID, O, V[Int]], D] =	{
		var cpt = 0
		var allModHaveConverged = false
		while( cpt < maxIter && ! allModHaveConverged ) {
			val centersInfo = obtainvalCentersInfo.map{ case (clusterID, (cardinality, preMode)) => 
				(
					clusterID,
					preMode.map( x => if( x * 2 >= cardinality ) 1 else 0 ).asInstanceOf[V[Int]],
					cardinality
				)
			}
			allModHaveConverged = checkIfConvergenceAndUpdateCenters(centersInfo, epsilon)
			cpt += 1
		}
		new KModesModel[ID, O, V[Int], Cz[ID, O, V[Int]], D](centers, metric)
	}
}

object KModes
{
	def run[
		ID: Numeric,
		O,
		V[Int] <: Seq[Int],
		Cz[ID, O, V <: Seq[Int]] <: BinaryClusterizable[ID, O, V, Cz[ID, O, V]]
	](
		@(transient @param) sc: SparkContext,
		data: RDD[Cz[ID, O, V[Int]]],
		k: Int,
		epsilon: Double,
		maxIter: Int,
		initializedCenters: mutable.HashMap[Int, V[Int]] = mutable.HashMap.empty[Int, V[Int]],
		persistanceLVL: StorageLevel = StorageLevel.MEMORY_ONLY
	)(implicit ct: ClassTag[Cz[ID, O, V[Int]]], ct2: ClassTag[V[Int]]): KModesModel[ID, O, V[Int], Cz[ID, O, V[Int]], Hamming[V[Int]]] = {
		val metric = new Hamming[V[Int]]
		val kmodes = new KModes(sc, data, k, epsilon, maxIter, metric, initializedCenters, persistanceLVL)
		val kModesModel = kmodes.run()
		kModesModel
	}
}