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

  def radial_factor(input, centroid, deviation) when (length(input) == length(centroid)) and is_number(deviation) do
    Utils.euclidean_distance(centroid, input)/deviation
  end

  def gaussian(input, centroid, deviation) do
    radius = radial_factor(input, centroid, deviation)
    numerator = -(:math.pow(radius, 2))
    denominator = 2
    :math.exp(numerator/denominator)
  end

  def linear(input) do
    input
  end

end

defmodule VecOps do
  def dot(inputs,weights) when length(inputs) == length(weights) do
    Enum.zip(inputs, weights) |> Enum.map(fn {a, b} -> a*b end) |> Enum.sum
  end

  def add(a,b) when length(a) == length(b) do
    Enum.zip(a,b) |> Enum.map(fn {a,b} -> a+b end)
  end

  def scale(a,b) when is_list(a) do
    Enum.map(a, fn x -> x * b end)
  end
end

defmodule KMeans do

  def run(dataset,k) do
    initial_centroids = init_centroids(dataset, k) 
    initial_clusters = assign_clusters(dataset, initial_centroids)
    run(dataset, initial_centroids,initial_clusters, 0)
  end

  defp run(dataset,centroids,clusters, count) when is_list(centroids) do
    if(count < 1000) do
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
    shape = hd(dataset) |> length
    for _ <- 1..k, do: for _ <- 1..shape, do: Utils.random_float(-50,50)
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
               |> VecOps.scale(1/length(cluster))
      ) 
    else
      labeled_centroids |> Enum.filter(fn cent -> elem(cent,1) == n end) |> hd |> elem(0)
    end)
  end
end

defmodule KNeighbors do

  def run(centroids) when is_list(centroids) do
    centroids |> Enum.zip(find_deviations(centroids))
  end

  def find_deviations(centroids) when is_list(centroids) do
    cen_mins = for cen <- centroids, do: (
      List.delete(centroids, cen)
      |> Enum.map(fn cen_2 -> Utils.euclidean_distance(cen, cen_2) end)
      |> Enum.sort
      |> Enum.take(2)
      |> List.to_tuple
    )
    #cen_mins |> IO.inspect
    for min <- cen_mins, do: :math.sqrt(elem(min,0)*elem(min,1))
  end
end

defmodule RadialNeuron do
  defstruct [:id, :center, :deviation]

  def init(id, center, deviation) do
    %RadialNeuron{id: id, center: center, deviation: deviation}
  end

  def activate(neuron, input)  do
    Utils.gaussian(input, neuron.center, neuron.deviation)
  end
end

defmodule OutputNeuron do
  defstruct [:id, :weights, :bias]

  def init(id, num_weights) when is_integer(num_weights) do
    w = for _ <- 1..num_weights, do: Utils.random_float(-5,5)
    b = Utils.random_float(0,1)
    %OutputNeuron{id: id, weights: w, bias: b}
  end

  def activate(neuron, input)  do
    VecOps.dot(neuron.weights, input) |> (Kernel.+neuron.bias) |> Utils.sigmoid
  end

  def update_weights(neuron, input, exp_out, act_out, rate) do
    delta = for x <- input, do: rate * (exp_out-act_out) * x
    nws = VecOps.add(neuron.weights, delta)
    %{neuron | weights: nws}
  end
end

defmodule RadialNet do
  defstruct [:radial_neurons, :output_neurons, learning_rate: 0.07]

  def init_radial_part(dataset, cluster_num) do
    clusters = KMeans.run(dataset,cluster_num)
    KNeighbors.run(clusters.centroids)
  end

  def init(radial_part, output_num) do
    num_clusters = hd(radial_part) |> elem(0) |> length |> IO.inspect
    radial_neurons = for id <- 0..length(radial_part)-1, do: (
    info = Enum.at(radial_part, id)
    %RadialNeuron{
        id: id, 
        center: info |> elem(0), 
        deviation: info |> elem(1)
      }
    )
    output_neurons = for n <- 0..output_num-1, do: OutputNeuron.init(n,num_clusters)
    %RadialNet{radial_neurons: radial_neurons, output_neurons: output_neurons}
  end
end

dataset = [[1,2,1],[2,2,2],[3,2,1],[0,0,0],[2,1,3],[4,5,2],[2,54,2],[1,1,-6],[-2,-2,-7]]
RadialNet.init_radial_part(dataset, 3) |> RadialNet.init(2)  |> IO.inspect
