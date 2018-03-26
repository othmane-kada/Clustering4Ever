package lipn.spartakus.core.math.distances

import _root_.scala.math.{pow, sqrt}

class Euclidean(root: Boolean) extends ContinuousDistances
{

	private def euclideanIntern(dot1: Seq[Double], dot2: Seq[Double]) =
	{
		var d = 0D
		for( i <- dot1.indices ) d += pow(dot1(i) - dot2(i), 2)
		d
	}

	/**
	  * The famous euclidean distance implemented in its fast mono thread scala version without SQRT part
	  * @return The Euclidean distance between dot1 and dot2
	  **/
	override def distance(dot1: Seq[Double], dot2: Seq[Double]): Double =
	{
		if( root ) sqrt(euclideanIntern(dot1, dot2))
		else euclideanIntern(dot1, dot2)
	}

	lazy val toStringRoot = if( root ) "with " else "without "

	override def toString = "Euclidean " + toStringRoot + "root applied"

}



