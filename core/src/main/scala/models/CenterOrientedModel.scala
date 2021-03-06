package org.clustering4ever.clustering.models
/**
 * @author Beck Gaël
 */
import scala.language.higherKinds
import org.clustering4ever.math.distances.{GSimpleVectorDistance, Distance, ContinuousDistance, BinaryDistance, MixedDistance}
import scala.collection.{mutable, immutable, GenSeq}
import org.clustering4ever.vectors.{GVector, GSimpleVector,ScalarVector, BinaryVector, MixedVector}
import org.clustering4ever.clusterizables.Clusterizable
/**
 * @tparam V
 * @tparam D
 */
trait CenterModel[V <: GVector[V], D <: Distance[V]] extends MetricModel[V, D] {
	/**
	 * Prototypes of clusters, ie elements which minimize distances to each members of their respective clusters
	 */
	val centers: immutable.HashMap[ClusterID, V]
	/**
	 * Time complexity O(c) with c the number of clusters
	 * @return the clusterID of nearest cluster center for a specific point
	 */
	final def centerPredict(v: V): ClusterID = centers.minBy{ case(_, centroid) => metric.d(centroid, v) }._1
}
/**
 * @tparam T
 * @tparam V
 * @tparam SV
 * @tparam D
 */
trait CenterModelSimpleV[T, V <: Seq[T], SV <: GSimpleVector[T, V, SV], D <: GSimpleVectorDistance[T, V, SV]] extends CenterModel[SV, D] {
	/**
	 * Time complexity O(c) with c the number of clusters
	 * @return the clusterID of nearest cluster center for a specific point
	 */
	final def centerPredict(v: V): ClusterID = centers.minBy{ case(_, centroid) => metric.d(centroid.vector, v) }._1	
}
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelReal[V <: Seq[Double], D <: ContinuousDistance[V]] extends CenterModelSimpleV[Double, V, ScalarVector[V], D]
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelBinary[V <: Seq[Int], D <: BinaryDistance[V]] extends CenterModelSimpleV[Int, V, BinaryVector[V], D]
/**
 * @tparam Vs
 * @tparam Vb
 * @tparam D
 */
trait CenterModelMixed[Vb <: Seq[Int], Vs <: Seq[Double], D <: MixedDistance[Vb, Vs]] extends CenterModel[MixedVector[Vb, Vs], D] {
	/**
	 * Time complexity O(c) with c the number of clusters
	 * @return the clusterID of nearest cluster center for a specific point
	 */
	final def centerPredict(v: (Vb, Vs)): ClusterID = centers.minBy{ case(_, centroid) => metric.d((centroid.binary, centroid.scalar), v) }._1
}
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelCz[V <: GVector[V], D <: Distance[V]] extends CenterModel[V, D] {
	/**
	 * Time complexity O(c) with c the number of clusters
	 * @return the clusterID of nearest cluster center for a specific point
	 */
	final def centerPredict[O, Cz[B, C <: GVector[C]] <: Clusterizable[B, C, Cz]](cz: Cz[O, V]): ClusterID = centerPredict(cz.v)

}
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelLocal[V <: GVector[V], D <: Distance[V]] extends CenterModel[V, D] {
	/**
	 * Time complexity O(n<sub>data</sub>.c) with c the number of clusters
	 * @return the input Seq with labels obtain via centerPredict method
	 */
	final def centerPredict[GS[X] <: GenSeq[X]](data: GS[V]): GS[(ClusterID, V)] = data.map( v => (centerPredict(v), v) ).asInstanceOf[GS[(ClusterID, V)]]
}
/**
 * @tparam T
 * @tparam V
 * @tparam SV
 * @tparam D
 */
trait CenterModelSimpleVLocal[T, V <: Seq[T], SV <: GSimpleVector[T, V, SV], D <: GSimpleVectorDistance[T, V, SV]] extends CenterModelSimpleV[T, V, SV, D] {
	/**
	 * Time complexity O(n<sub>data</sub>.c) with c the number of clusters
	 * @return the input Seq with labels obtain via centerPredict method
	 */
	final def centerPredict[GS[X] <: GenSeq[X]](data: GS[V])(implicit d1: DummyImplicit, d2: DummyImplicit): GS[(ClusterID, V)] = data.map( v => (centerPredict(v), v) ).asInstanceOf[GS[(ClusterID, V)]]
}
/**
 * @tparam Vb
 * @tparam Vs
 * @tparam D
 */
trait CenterModelMixedLocal[Vb <: Seq[Int], Vs <: Seq[Double], D <: MixedDistance[Vb, Vs]] extends CenterModelMixed[Vb, Vs, D] {
	/**
	 * Time complexity O(n<sub>data</sub>.c) with c the number of clusters
	 * @return the input Seq with labels obtain via centerPredict method
	 */
	final def centerPredict[GS[X] <: GenSeq[X]](data: GS[(Vb, Vs)])(implicit d1: DummyImplicit, d2: DummyImplicit): GS[(ClusterID, (Vb, Vs))] = data.map( v => (centerPredict(v), v) ).asInstanceOf[GS[(ClusterID, (Vb, Vs))]]
}
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelLocalReal[V <: Seq[Double], D <: ContinuousDistance[V]] extends CenterModelSimpleVLocal[Double, V, ScalarVector[V], D]
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelLocalBinary[V <: Seq[Int], D <: BinaryDistance[V]] extends CenterModelSimpleVLocal[Int, V, BinaryVector[V], D]
/**
 * @tparam V
 * @tparam D
 */
trait CenterModelLocalCz[V <: GVector[V], D <: Distance[V]] extends CenterModelLocal[V, D] with CenterModelCz[V, D] {
	/**
	 * Time complexity O(n<sub>data</sub>.c) with c the number of clusters
	 * @return the input Seq with labels obtain via centerPredict method
	 */
	final def centerPredict[O, Cz[B, C <: GVector[C]] <: Clusterizable[B, C, Cz], GS[X] <: GenSeq[X]](data: GS[Cz[O, V]])(implicit d: DummyImplicit): GS[Cz[O, V]] = {
		data.map( cz => cz.addClusterIDs(centerPredict(cz)) ).asInstanceOf[GS[Cz[O, V]]]
	}
}