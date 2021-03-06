 package org.clustering4ever.scala.clustering.tensor
/**
 * @author ANDRIANTSIORY Dina Faneva, Beck Gaël
 */
import scala.collection.mutable
import breeze.linalg.svd.SVD
import breeze.stats.mean
import breeze.linalg._
import scala.math._

class RecursiveBiclusters(val l1: Array[Int], val l2: Array[Int]) {

  def oneBicluster(k1: Int, k2: Int, tensor1: mutable.ArrayBuffer[DenseMatrix[Double]]) = {
    
    val m = tensor1.length
    val n1 = tensor1.head.rows
    val n2 = tensor1.head.cols
    val timeColumn = DenseMatrix.zeros[Double](m, n2)  
    val timeRow = DenseMatrix.zeros[Double](m, n1)

    @annotation.tailrec
    def matriceColumnSet(t: mutable.ArrayBuffer[DenseMatrix[Double]], m: DenseMatrix[Double], c: DenseMatrix[Double], i: Int, j: Int , k: Int): DenseMatrix[Double] = {
      if(j < t.head.cols && k < t.length) {
        m(k,j) = t(k)(i,j)
        matriceColumnSet(t, m, c, i, j, k + 1)
      }
      else if(k == t.length && j < t.head.cols) {
        matriceColumnSet(t, m, c, i, j + 1, 0)
      }

      else if(i < t.head.rows - 1) {
        c += cov(m)
        matriceColumnSet(t, m, c, i + 1, 0, 0)
      }
      else {
        c += cov(m)
      }
    }

    @annotation.tailrec
    def matriceRowSet(t: mutable.ArrayBuffer[DenseMatrix[Double]], m: DenseMatrix[Double], c: DenseMatrix[Double], i: Int, j: Int , k: Int): DenseMatrix[Double] = {
      if(i < t.head.rows && k < t.length) {
        m(k,i) = t(k)(i,j)
        matriceRowSet(t, m, c, i, j, k + 1)
      }
      else if(k == t.length && i < t.head.rows) {
        matriceRowSet(t, m, c, i + 1, j, 0)
      }
      else if(j < t.head.cols - 1) {
        c += cov(m)
        matriceRowSet(t, m, c, 0, j + 1, 0)
      }
      else {
        c += cov(m)
      }
    }

    val columnMatrix = matriceColumnSet(tensor1, timeColumn, DenseMatrix.zeros[Double](n2, n2), 0, 0, 0 )
    val svd.SVD(u1, eigValue, eigVector) = svd(columnMatrix)
    val columnEigvalue = eigValue.toArray
    val columnEigvector = eigVector.t
         
 
    val rowMatrix = matriceRowSet(tensor1, timeRow, DenseMatrix.zeros[Double](n1, n1), 0, 0, 0 )
    val svd.SVD(u2,eigValue2,eigVector2) = svd(rowMatrix)
    val rowEigvalue = eigValue2.toArray
    val rowEigvector = eigVector2.t
   
    def geTTheTopkIndices(m: DenseMatrix[Double], k: Int): Array[Int] = {
      m(::,0).map(abs(_)).toArray.zipWithIndex.sortWith((x, y) => x._1 > y._1).take(k).map(_._2)
    }
  
    val row = geTTheTopkIndices(rowEigvector, k1)
    val column = geTTheTopkIndices(columnEigvector, k2)

    // println("\n\n bicluster number "+": \n rows "+ row.toList + "\n column "+ column.toList )

    def annulation(t: mutable.ArrayBuffer[DenseMatrix[Double]], jf: Array[Int], js: Array[Int]): mutable.ArrayBuffer[DenseMatrix[Double]] = { 
      for (k <- 0 until t.length) {
        for (i <- jf) {
          for (j <- js) {
                t(k)(i, j) = 0 
          }
        }
      }
      t
    }

    val tensorRemain = annulation(tensor1, row, column)

    (tensorRemain, Array(row, column) )
  }
  

  def fit(data: mutable.ArrayBuffer[DenseMatrix[Double]]) = {

    var r1 = mutable.ListBuffer[mutable.ArrayBuffer[DenseMatrix[Double]]]()
    r1 += data
    var result = mutable.ListBuffer[Array[Array[Int]]]()

    
    for(nombre <- 0 until l1.length) {
      val (t1, re1) = oneBicluster(l1(nombre), l2(nombre), r1(nombre))
      r1 += t1
      result += re1
    }

    result
  }
}  
/**
 *
 */
object RecursiveBiclusters {
  
  def train(k1: Array[Int], k2: Array[Int], data: mutable.ArrayBuffer[DenseMatrix[Double]]) = (new RecursiveBiclusters(k1, k2)).fit(data)
  
}