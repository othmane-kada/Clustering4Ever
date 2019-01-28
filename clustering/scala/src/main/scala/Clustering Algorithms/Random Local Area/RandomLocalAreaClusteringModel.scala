package org.clustering4ever.scala.clustering.rla
/**
 * @author Beck Gaël
 */
import scala.language.higherKinds
import org.clustering4ever.clustering.GenericClusteringModel
import scala.collection.{immutable, GenSeq}
import org.clustering4ever.math.distances.{Distance, ContinuousDistance, BinaryDistance, MixtDistance}
import org.clustering4ever.clusterizables.Clusterizable
import org.clustering4ever.clustering.models.{CenterModelLocalReal, CenterModelLocalBinary, CenterModelMixtLocal, CenterModelLocalCz, KnnModelModelLocalCz, KnnModelModel, KnnModelModelReal, KnnModelModelBinary, KnnModelModelMixt}
import org.clustering4ever.vectors.{GVector, ScalarVector, BinaryVector, MixtVector}
import org.clustering4ever.clustering.ClusteringModelLocal
/**
 *
 */
trait RLAModelAncestor[ID, O, V <: GVector[V], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D <: Distance[V], GS[X] <: GenSeq[X], +CA <: RLAArgsAncestor[V, D]] extends CenterModelLocalCz[ID, O, V, Cz, D] with KnnModelModelLocalCz[ID, O, V, Cz, D] with ClusteringModelLocal[ID, O, V, Cz, GS, CA] {
	/**
	 *
	 */
	def obtainClustering(data: GS[Cz[ID, O, V]]): GS[Cz[ID, O, V]] = centerPredict(data)
}
/**
 *
 */
case class RLAModel[ID, O, V <: GVector[V], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: GVector[X]] <: Distance[X], GS[X] <: GenSeq[X]](val centers: immutable.HashMap[Int, V], val metric: D[V], val args: RLAArgs[V, D]) extends RLAModelAncestor[ID, O, V, Cz, D[V], GS, RLAArgs[V, D]] with KnnModelModel[V, D[V]]
/**
 *
 */
case class RLAModelScalar[ID, O, V <: Seq[Double], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: Seq[Double]] <: ContinuousDistance[X], GS[X] <: GenSeq[X]](val centers: immutable.HashMap[Int, ScalarVector[V]], val metric: D[V], val args: RLAArgsScalar[V, D]) extends RLAModelAncestor[ID, O, ScalarVector[V], Cz, D[V], GS, RLAArgsScalar[V, D]]  with CenterModelLocalReal[V, D[V]] with KnnModelModelReal[V, D[V]]
/**
 *
 */
case class RLAModelBinary[ID, O, V <: Seq[Int], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: Seq[Int]] <: BinaryDistance[X], GS[X] <: GenSeq[X]](val centers: immutable.HashMap[Int, BinaryVector[V]], val metric: D[V], val args: RLAArgsBinary[V, D]) extends RLAModelAncestor[ID, O, BinaryVector[V], Cz, D[V], GS, RLAArgsBinary[V, D]] with CenterModelLocalBinary[V, D[V]] with KnnModelModelBinary[V, D[V]]
/**
 *
 */
case class RLAModelMixt[ID, O, Vb <: Seq[Int], Vs <: Seq[Double], Cz[X, Y, Z <: GVector[Z]] <: Clusterizable[X, Y, Z, Cz], D[X <: Seq[Int], Y <: Seq[Double]] <: MixtDistance[X, Y], GS[X] <: GenSeq[X]](val centers: immutable.HashMap[Int, MixtVector[Vb, Vs]], val metric: D[Vb, Vs], val args: RLAArgsMixt[Vb, Vs, D]) extends RLAModelAncestor[ID, O, MixtVector[Vb, Vs], Cz, D[Vb, Vs], GS, RLAArgsMixt[Vb, Vs, D]] with CenterModelMixtLocal[Vb, Vs, D[Vb, Vs]] with KnnModelModelMixt[Vb, Vs, D[Vb, Vs]]