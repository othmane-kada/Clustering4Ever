package org.clustering4ever.clustering.epsilonproximity.rdd
/**
 * @author Beck Gaël
 */
import scala.collection.{mutable, immutable}
/**
 * Find isolate graph
 */
object GatherClustersWithSharedDots {

	final def generateNodesAndPairsNew(seq: Seq[Seq[Int]]): (immutable.HashSet[Int], immutable.HashMap[Int, immutable.HashSet[Int]]) = {
		val nodes = immutable.HashSet(seq.flatten:_*)
		val allPairs = immutable.HashMap(seq.flatMap( ss => if(ss.size != 1) ss.map( node => (node, immutable.HashSet(ss.filter(_ != node):_*)) ) else Seq((ss.head, immutable.HashSet(ss.head))) ):_*)
		(nodes, allPairs)
	}
	/**
	 * @return List of dots which are in the same cluster
	 */
	final def reduceByGatheringClusters(nodes: immutable.HashSet[Int], neighbours: immutable.HashMap[Int, immutable.HashSet[Int]]): List[List[Int]] = {

		val visited = mutable.HashMap.empty[Int, Int]

	    def depthFirstTraverseFunctional(node: Int, clusterID: Int): Unit = {

	      val nodeToExplore = immutable.HashSet(node)

	      def obtainUnvisitedNeihbors(hs: immutable.HashSet[Int]) = {
	        hs.flatMap{ n =>
	        	val unvisited = neighbours(n).filter( n => ! visited.contains(n) )
	        	visited ++= unvisited.map( uv => (uv, clusterID) )
	        	unvisited
	        }
	      }
	      
	      @annotation.tailrec
	      def go(hs: immutable.HashSet[Int]): immutable.HashSet[Int] = if(!hs.isEmpty) go(obtainUnvisitedNeihbors(hs)) else hs

	      go(nodeToExplore)
	    }

		var clusterID = 0

	    nodes.foreach( n =>
	      if(!visited.contains(n)) {
	        visited += ((n, clusterID))
	        depthFirstTraverseFunctional(n, clusterID)
	        clusterID += 1
	      }
	    )

		val labeledNodes = nodes.toList.map( n => (n, visited(n)) )
		val labels = labeledNodes.map(_._2)
		val returnReduce = labels.map( l => labeledNodes.collect{ case (n, cID) if cID == l => n } )

		returnReduce
	}

}