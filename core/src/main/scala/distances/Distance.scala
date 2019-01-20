package org.clustering4ever.math.distances
/**
 * @author Beck Gaël
 */
import scala.language.higherKinds
import org.clustering4ever.clusterizables.Clusterizable
import org.clustering4ever.vectors.{GVector, ScalarVector, BinaryVector, MixtVector}
import org.clustering4ever.types.MetricIDType._
/**
 *
 */
trait MetricArgs extends Serializable
/**
 * Most general notion of Distance, taking two object of type O and returning a Double
 */
trait GenericDistance[O] extends Serializable {
// trait Distance[O, MA <: MetricArgs] extends Serializable {
	// val metricArgs: MA
	/**
	 *
	 */
	def d(o1: O, o2: O): Double
	/**
	 *
	 */
	val id: MetricID
}
/**
 * The EmptyDistance for algorithm which doesn't require any distances
 */
object EmptyDistance extends Distance[Nothing] {
	/**
	 *
	 */
	def d(o1: Nothing, o2: Nothing): Double = 0D
	/**
	 *
	 */
	val id = 0
}
/**
 * Distance trait for applicable on GVector descendant like objects
 */
trait Distance[V <: GVector[V]] extends GenericDistance[V]
/**
 *
 */
trait ContinuousDistance[V <: Seq[Double]] extends Distance[ScalarVector[V]] {
	/**
	 * Distance on raw vector nature
	 */
	def d(v1: V, v2: V): Double
}
/**
 *
 */
trait BinaryDistance[V <: Seq[Int]] extends Distance[BinaryVector[V]] {
	/**
	 * Distance on raw vector nature
	 */
	def d(v1: V, v2: V): Double
}
/**
 *
 */
trait MixtDistance[Vb <: Seq[Int], Vs <: Seq[Double]] extends Distance[MixtVector[Vb, Vs]] {
	/**
	 * Distance on raw vector nature
	 */
	// def d(v1: (Vb, Vs), v2: (Vb, Vs)): Double
}
/**
 *
 */
trait RawContinuousDistance[V <: Seq[Double]] extends GenericDistance[V]
/**
 *
 */
trait RawBinaryDistance[V <: Seq[Int]] extends GenericDistance[V]