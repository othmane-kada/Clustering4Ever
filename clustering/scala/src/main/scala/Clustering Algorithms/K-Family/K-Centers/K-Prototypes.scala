package org.clustering4ever.clustering.kcenters.scala
/**
 * @author Beck Gaël
 */
import scala.language.higherKinds
import scala.reflect.ClassTag
import scala.collection.{immutable, GenSeq}
import scala.util.Random
import org.clustering4ever.math.distances.{MixtDistance, Distance}
import org.clustering4ever.math.distances.mixt.HammingAndEuclidean
import org.clustering4ever.clusterizables.{Clusterizable, EasyClusterizable}
import org.clustering4ever.util.ScalaCollectionImplicits._
import org.clustering4ever.vectors.{GVector, MixtVector}
/**
 *
 */
case class KPrototypes[ID, O, Vb <: Seq[Int], Vs <: Seq[Double], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: Seq[Int], Y <: Seq[Double]] <: MixtDistance[X, Y], GS[X] <: GenSeq[X]](val args: KPrototypesArgs[Vb, Vs, D]) extends KCentersAncestor[ID, O, MixtVector[Vb, Vs], Cz, D[Vb, Vs], GS, KPrototypesArgs[Vb, Vs, D], KPrototypesModel[ID, O, Vb, Vs, Cz, D, GS]] {

	def run(data: GS[Cz[ID, O, MixtVector[Vb, Vs]]]): KPrototypesModel[ID, O, Vb, Vs, Cz, D, GS] = KPrototypesModel(obtainCenters(data), args.metric, args)

	// def updateArgs(newArgs: KPrototypesArgs[Vb, Vs, D]): KPrototypes[ID, O, Vb, Vs, Cz, D, GS] = {
	// 	newArgs.obtainAlgorithm[ID, O, Cz, GS](None)
	// }

}
/**
 * The famous K-Prototypes using a user-defined dissmilarity measure.
 * @param data :
 * @param k : number of clusters
 * @param epsilon : minimal threshold under which we consider a centroid has converged
 * @param maxIterations : maximal number of iteration
 * @param metric : a defined dissimilarity measure
 */
object KPrototypes {
	/**
	 * Run the K-Prototypes with any mixt distance
	 */
	def run[ID, O, Vb <: Seq[Int], Vs <: Seq[Double], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: Seq[Int], Y <: Seq[Double]] <: MixtDistance[X, Y], GS[X] <: GenSeq[X]](
		data: GS[Cz[ID, O, MixtVector[Vb, Vs]]],
		k: Int,
		metric: D[Vb, Vs],
		maxIterations: Int,
		epsilon: Double,
		initializedCenters: immutable.HashMap[Int, MixtVector[Vb, Vs]] = immutable.HashMap.empty[Int, MixtVector[Vb, Vs]]
	): KPrototypesModel[ID, O, Vb, Vs, Cz, D, GS] = {
		
		val kPrototypesAlgorithm = new KPrototypes[ID, O, Vb, Vs, Cz, D, GS](KPrototypesArgs(k, metric, epsilon, maxIterations, initializedCenters))
		kPrototypesAlgorithm.run(data)
	
	}
}