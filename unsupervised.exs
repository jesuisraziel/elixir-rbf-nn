defmodule Utils do
  def random_float(lower,upper) do
    :rand.uniform() * (lower - upper) + upper
  end

  def euclidean_distance(a,b) when length(a) == length(b) do 
    Enum.zip(a,b)
    |> Enum.reduce(0,fn {an, bn}, acc -> :math.pow(bn-an,2)+acc end) 
    |> :math.sqrt
  end

  def add(list1, list2) when is_list(list1) and is_list(list2) do
    Enum.zip(list1, list2) |> Enum.map(fn {x,y} -> x+y end)
  end

  def scale(list, scalar) when is_list(list) do
    Enum.map(list, fn elem -> elem * scalar end)
  end

end


  defmodule KMeans do

    def run(dataset,k) do
      initial_centroids = init_centroids(dataset, k) 
      initial_clusters = assign_clusters(dataset, initial_centroids)
      run(dataset, initial_centroids,initial_clusters, 0)
    end

    defp run(dataset,centroids,clusters, count) when is_list(centroids) do
      if(count < 5) do
        new_centroids = update_centroids(clusters, centroids)
        new_clusters = assign_clusters(dataset, new_centroids)
        run(dataset, new_centroids, new_clusters, count+1); 
      else
        k = length(centroids)
        %{centroids: centroids, clustered_data: clusters}
      end
    end
    
    defp init_centroids(dataset,k) when is_list(dataset) do
      #Select a random dataset point to act as the centroid
      for _ <- 1..k, do: Enum.random(dataset)
    end
    
    defp assign_clusters(dataset, centroids) do
      for data <- dataset, do: 
      {data,  centroids 
        |> Enum.map(fn centroid -> Utils.euclidean_distance(centroid, data) end) 
        |> Enum.with_index 
        |> Enum.min 
        |> elem(1)
      }
    end

    defp update_centroids(labeled_dataset, centroids) do
      labeled_centroids = centroids |> Enum.with_index
      k = length(labeled_centroids)
      for n <- 0..k-1, do: (
      cluster = Enum.filter(labeled_dataset, fn data -> elem(data,1) == n end)
      unless cluster == [] do (
        initial_accumulator = List.duplicate(0, cluster |> hd |> elem(0) |> length)
        Enum.reduce(cluster, initial_accumulator, fn datapoint, acc -> Utils.add(acc, elem(datapoint,0)) end) 
               |> Utils.scale(1/length(cluster))
        ) 
        else
          labeled_centroids |> Enum.filter(fn cent -> elem(cent,1) == n end) |> hd |> elem(0)
        end)
    end
  end


dataset_a = [[1,2,1],[2,2,2],[3,2,1],[0,0,0],[2,1,3],[4,5,2],[2,54,2],[1,1,-6],[-2,-2,-7]]
KMeans.run(dataset_a, 3) |> IO.inspect
