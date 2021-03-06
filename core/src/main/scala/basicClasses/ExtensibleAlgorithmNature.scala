package org.clustering4ever.extensibleAlgorithmNature
/**
 *
 */
trait ClusteringAlgorithmNature extends Serializable
/**
 *
 */
final case object JenksNaturalBreaks extends ClusteringAlgorithmNature
/**
 *
 */
final case object TensorBiclustering extends ClusteringAlgorithmNature
/**
 *
 */
final case object KCenters extends ClusteringAlgorithmNature
/**
 *
 */
final case object KMeans extends ClusteringAlgorithmNature
/**
 *
 */
final case object KModes extends ClusteringAlgorithmNature
/**
 *
 */
final case object KPrototypes extends ClusteringAlgorithmNature
/**
 *
 */
final case object GaussianMixtures extends ClusteringAlgorithmNature
/**
 *
 */
final case object RLA extends ClusteringAlgorithmNature
/**
 *
 */
final case object RLAScalar extends ClusteringAlgorithmNature
/**
 *
 */
final case object RLABinary extends ClusteringAlgorithmNature
/**
 *
 */
final case object RLAMixt extends ClusteringAlgorithmNature
/**
 *
 */
final case object EpsilonProximity extends ClusteringAlgorithmNature
/**
 *
 */
final case object EpsilonProximityScalar extends ClusteringAlgorithmNature
/**
 *
 */
final case object EpsilonProximityBinary extends ClusteringAlgorithmNature
/**
 *
 */
final case object EpsilonProximityMixt extends ClusteringAlgorithmNature
/**
 *
 */
trait PreprocessingAlgorithmNature extends Serializable
/**
 *
 */
final case object GradientAscent extends PreprocessingAlgorithmNature