val clusteringName = "Clustering4Ever"

name := clusteringName

exportJars := true

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) => clusteringName + "." + artifact.extension }