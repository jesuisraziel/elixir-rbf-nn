defmodule FileUtils do
  @moduledoc """
    This module handles miscellaneous file operations, such as writing from CSV files.
  """
  @doc """
    Reads a CSV file line by line.

    Returns a list containing newline-terminated string representations of each line in the file.
  """
  def read_csv(path) do
    File.stream!(path) |> Enum.to_list
  end

  @doc """
    Parses a list whose sole element is a comma separated list of numbers, like the ones returned by read_csv

    Returns a list of numbers.
  """

  def parse_csv_line(list) do
    hd(list)
    |> String.replace("\n", "")
    |> String.split(",")
    |> Enum.map(fn elem -> String.to_integer(elem) end)
  end

  @doc """
    Recursively parses the output of read_csv and transform it into a usable form.

    We assume the file contains an even number of lines. The pattern goes input value -> input label, and
    then repeats.

    Returns a list of tuples in which the first element is the input, and the second one is its expected
    value (label)
  """

  def parse_dataset_csv(list) when list == [] do
    list
  end

  def parse_dataset_csv(list) do
    [input, label | rest] = list

    parsed_input = input
    |> String.replace("\n", "")
    |> String.split(",")
    |> Enum.map(fn elem -> String.to_integer(elem) end)

    parsed_label = label
    |> String.replace("\n", "")
    |> String.split(",")
    |> Enum.map(fn elem -> String.to_integer(elem) end)

    [{parsed_input, parsed_label}|parse_dataset_csv(rest)]
  end

  @doc """
    Converts an Erlang term to binary and saves it to a file.
  """
  def serialize(element, path) do
    bin = :erlang.term_to_binary(element)
    File.write!(path, bin)
  end

  @doc """
    Reads Erlang binary datafrom disk and converts it back to a term.
  """
  def deserialize(path) do
    File.read!(path) |> :erlang.binary_to_term
  end
end

defmodule MathUtils do
  @moduledoc """
    This module handles miscellanous math operations pertinent to the program.
  """
  @doc """
    Generates a random number between an upper and lower bound.

    Returns a floating point value.
  """
  def random_float(lower,upper) do
    :rand.uniform() * (lower - upper) + upper
  end

  @doc """
    Calculates the euclidean distance between two vectors of the same size.

    Returns a floating point value.
  """

  def euclidean_distance(a,b) when length(a) == length(b) do
    Enum.zip(a,b)
    |> Enum.reduce(0,fn {an, bn}, acc -> :math.pow(bn-an,2)+acc end)
    |> :math.sqrt
  end
end

defmodule NetUtils do
  @moduledoc """
    This module handles operations that are useful within a Radial Basis Function Neural Network,
    mainly computing the value of several activation functions.
  """

  @doc """
    Computes the "radial factor" of the gaussian function. That is, the euclidean distance
    between a vector and a certain centroid (or point in space), divided by the centroid's
    deviation.

    Returns a floating point value.
  """
  def radial_factor(input, centroid, deviation)
    when (length(input) == length(centroid))
    and is_number(deviation)
  do
    MathUtils.euclidean_distance(centroid, input)/deviation
  end

  @doc """
    Computes the gaussian function given an input, a centroid and its deviation.

    Returns a floating point value.
  """
  def gaussian(input, centroid, deviation) do
    radius = NetUtils.radial_factor(input, centroid, deviation)
    numerator = -(:math.pow(radius, 2))
    denominator = 2
    :math.exp(numerator/denominator)
  end

  @doc """
    Computes the identity function.

    Returns the input.
  """
  def linear(input) do
    input
  end
end

defmodule VecOps do
  @doc """
    Computes the dot product between two vectors of the same size.

    Returns the dot product.
  """
  def dot(inputs,weights)
    when length(inputs) == length(weights)
  do
    Enum.zip(inputs, weights) |> Enum.map(fn {a, b} -> a*b end) |> Enum.sum
  end

  @doc """
    Performs element-wise addition of every element in a vector.

    Returns the sum betweeen vectors.
  """
  def add(a,b)
    when length(a) == length(b)
  do
    Enum.zip(a,b) |> Enum.map(fn {a,b} -> a+b end)
  end

  @doc """
    Performs element-wise scaling of every element in a vector.

    Returns the scaled vector.
  """
  def scale(a,b)
    when is_list(a)
  do
    Enum.map(a, fn x -> x * b end)
  end
end

defmodule KMeans do
  @moduledoc """
    This module implements the K-Means clustering algorithm.
  """

  @doc """
    Performs the K-Means clustering algorithm.

    We find the initial centroids, then we assign clusters to each one of them, and we
    repeat until the desired number of iterations is reached. Most of the functionality
    is achieved by means of utility functions, which we individually document.

    Returns a map containing the centroids of each cluster, as well as the clustered
    data itself.
  """
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
      %{centroids: centroids, clustered_data: clusters}
    end
  end

  @doc """
    Determines what the initial centroids will be in the K-Means algorithm.

    We do this by generating a random vector within the input space.

    Returns a list of initial centroids.
  """
  defp init_centroids(dataset,k) when is_list(dataset) do
    #Select a random dataset point to act as the centroid
    shape = hd(dataset) |> length
    for _ <- 1..k, do: (
      for _ <- 1..shape, do: MathUtils.random_float(-2,2)
    )
  end

  @doc """
    Assign each datapoint to a single centroid, as per the requirements of the
    algorithm (the chosen cluster is the one closest to the datapoint.)

    Returns a list of tuples containing each datapoint, as well as its assigned cluster.
  """

  defp assign_clusters(dataset, centroids) do
    for data <- dataset, do:
    {data,  centroids
      |> Enum.map(fn centroid -> MathUtils.euclidean_distance(centroid, data) end)
      |> Enum.with_index
      |> Enum.min
      |> elem(1)
    }
  end

@doc """
  Updates centroids within the K-Means clustering algorithm.

  If a centroid has at least one datapoint assigned to it, we take the mean of all datapoints
  assigned to it and set it as the new centroid. Else, we keep it as-is.

  Returns the list of updated centroids.
"""
  defp update_centroids(labeled_dataset, centroids) do
    labeled_centroids = centroids |> Enum.with_index
    k = length(labeled_centroids)
    for n <- 0..k-1, do: (
      cluster = Enum.filter(labeled_dataset, fn data -> elem(data,1) == n end)
      unless cluster == [] do (
        initial_accumulator = List.duplicate(0, cluster |> hd |> elem(0) |> length)
        Enum.reduce(cluster, initial_accumulator, fn datapoint, acc -> VecOps.add(acc, elem(datapoint,0)) end)
        |> VecOps.scale(1/length(cluster))
      )
    else
      labeled_centroids |> Enum.filter(fn cent -> elem(cent,1) == n end) |> hd |> elem(0)
    end)
  end
end

defmodule KNeighbors do
  @moduledoc """
    This module implements part of the K-Nearest Neighbors algorithm. Specifically, we use
    it only to calculate the deviation of each centroid, rather than using it to assign a
    new datapoint to an existing cluster, as is traditional.
  """

  @doc """
    Runs the K-Nearest Neighbors algorithm.

    Returns a list of tuples containing each centroid alongside its deviation.
  """
  def run(centroids) when is_list(centroids) do
    centroids |> Enum.zip(find_deviations(centroids))
  end

  @doc """
    Finds each centroid's deviation.

    We do this by calculating the distance between the centroid and its two nearest
    neighbord, multiplying them and taking the square root.

    Returns a list of tuples containing each centroid alongside its deviation.
  """
  def find_deviations(centroids) when is_list(centroids) do
    cen_mins = for cen <- centroids, do: (
      List.delete(centroids, cen)
      |> Enum.map(fn cen_2 -> MathUtils.euclidean_distance(cen, cen_2) end)
      |> Enum.sort
      |> Enum.take(2)
      |> List.to_tuple
    )
    for min <- cen_mins, do: :math.sqrt(elem(min,0)*elem(min,1))
  end
end

defmodule RadialNeuron do
  @moduledoc """
    This module defines a structure which correlates to a neuron with a radial activation
    function, as well as methods that are important when working with it.
  """
  defstruct [:id, :center, :deviation]

  @doc """
    Initializes a radial neuron.

    Returns an initialized radial neuron.
  """
  def init(id, center, deviation) do
    %RadialNeuron{id: id, center: center, deviation: deviation}
  end

  @doc """
    Activates a radial neuron.

    Returns the neuron's activation.
  """
  def activate(neuron, input)  do
    NetUtils.gaussian(input, neuron.center, neuron.deviation)
  end
end

defmodule OutputNeuron do
  @moduledoc """
    This module defines a structure which correlates to a neuron with a linear activation
    function, as well as methods that are important when working with it.
  """
  defstruct [:id, :weights, :bias]

  @doc """
    Initalizes an output neuron with a random bias and random weights.

    Return the initialized neuron.
  """
  def init(id, num_weights) when is_integer(num_weights) do
    w = for _ <- 1..num_weights, do: MathUtils.random_float(-2,2)
    b = MathUtils.random_float(0,1)
    %OutputNeuron{id: id, weights: w, bias: b}
  end

  @doc """
    Activates an output neuron.

    Returns the neuron's activation, rounded up to the nearest integer.
  """
  def activate(neuron, input)  do
    VecOps.dot(neuron.weights, input) |> (Kernel.+neuron.bias) |> NetUtils.linear |> Kernel.round()
  end

  @doc """
    Updates an output neuron's bias and weights by means of the delta rule.

    Returns the updated neuron.
  """
  def update_weights(neuron, input, exp_out, act_out, rate) do
    delta = for x <- input, do: rate * (exp_out-act_out) * x
    nws = VecOps.add(neuron.weights, delta)
    updated_weights = %{neuron | weights: nws}
    bias_adjustment = rate * (exp_out - act_out) * 1
    new_bias = neuron.bias + bias_adjustment
    %{updated_weights | bias: new_bias}
  end
end

defmodule RadialNet do
  @moduledoc """
    This module contains methods to work with radial basis function neural networks,
    as well as as a struct that represents an instance of one.
  """
  defstruct [:radial_neurons, :output_neurons, learning_rate: 0.07]

  @doc """
    Initializes the hidden layer of the neural network.

    Returns a list of `cluster_num` tuples, containing a centroid and
    its deviation.
  """
  def init_radial_part(dataset, cluster_num) do
    clusters = KMeans.run(dataset,cluster_num) #|> IO.inspect
    KNeighbors.run(clusters.centroids)
  end

  @doc """
    Fully initializes a neural network, given the data needed for
    its hidden layer.

    Returns an initialized network.
  """
  def init(radial_part, output_num) do
    num_clusters = length(radial_part)
    radial_neurons = for id <- 0..num_clusters-1, do: (
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

  @doc """
    Activates the network's hidden layer.

    Returns the hidden layer's output.
  """
  def activate_hidden_layer(net, input) do
    for neuron <- net.radial_neurons, do: (
      RadialNeuron.activate(neuron, input)
    )
  end

  @doc """
    Processes an output.

    Returns the network's prediction.
  """
  def predict(net, input) do
    hidden_layer_output = activate_hidden_layer(net, input)
    for neuron <- net.output_neurons, do: OutputNeuron.activate(neuron, hidden_layer_output)
  end

  @doc """
    Trains the neural network on a given dataset for a specified number of epochs.

    This is a recursive function which depends on several helper functions.

    Returns the trained network.
  """
  def train_on_dataset(net, labeled_dataset,example_count, epochs) do
    if(epochs == 0) do
      net
    else
      IO.puts("Epoca #{to_string(epochs)}")
      updated_net = train_on_dataset(net, labeled_dataset)
      shuffled_dataset = Enum.shuffle(labeled_dataset)
      train_on_dataset(updated_net, shuffled_dataset, example_count, epochs-1)
    end
  end

  def train_on_dataset(net, labeled_dataset) when labeled_dataset == [] do
    net
  end

  def train_on_dataset(net, labeled_dataset) do
    [example | remaining_data] = labeled_dataset
    updated_net = train_on_example(net, example)
    train_on_dataset(updated_net,remaining_data)
  end

  @doc """
    Trains the neural network based on a given example.

    We use stochastic gradient descent, so the weights connecting the hidden and output
    layer are adjusted after each and every training example.

    We calculate both the network's final output and the hidden layer output because
    we're adjusting the output layer: we can't work directly with the network's input
    as it's been transformed by the hidden layer.

    Returns the adjusted network.
  """

  def train_on_example(net, labeled_input) do
    {input, label} = labeled_input
    hidden_layer_output = activate_hidden_layer(net, input)
    output = RadialNet.predict(net, input)
    to_zip = [net.output_neurons,label,output]
    zipped_outputs = List.zip(to_zip)
    updated_neurons = for {neuron, expected, actual} <- zipped_outputs, do: (
      OutputNeuron.update_weights(neuron, hidden_layer_output, expected, actual, net.learning_rate)
    )
    %{net | output_neurons: updated_neurons}
  end

  @doc """
    Calculates the network's accuracy over a given testing set.

    Returns a map with the amount of training examples, how many correct guesses were
    made, and the network's accuracy.
  """
  def calculate_accuracy(net, labeled_dataset) do
    data = Enum.map(labeled_dataset, fn item -> elem(item,0) end)
    labels = Enum.map(labeled_dataset, fn item -> elem(item,1) end)
    outputs = Enum.map(data, fn datum -> RadialNet.predict(net, datum) end)
    zipped_outputs = Enum.zip(labels,outputs)
    example_count = length(labeled_dataset)
    hits = Enum.reduce(zipped_outputs, 0, fn {exp, act}, acc -> if exp == act do acc + 1 else acc + 0 end end)
    accuracy = hits/length(labeled_dataset)
    %{example_count: example_count, hits: hits, accuracy: accuracy}
  end
end

defmodule ProgramUtils do
  @moduledoc """
    This module provides utilities when running the program from within the command
    line.
  """

  @doc """
    Interprets the neural network's output.

    Returns a human-readable response.
  """
  def interpret(output) do
    output |> IO.inspect
    sections = ["Alteraciones severas en la interacción social recíproca: ",
    "Patrón / es de intereses restringidos y absorbentes: ",
    "Imposición de rutinas, rituales e intereses: ",
    "Peculiaridades del habla y el lenguaje: ",
    "Problemas de comunicación no verbal: ",
    "Torpeza motora: "]
    results = Enum.map(output, fn out -> if out == 1 do "Criterio cumplido" else "------" end end)
    Enum.zip(sections, results)
    |> Enum.map(fn {sec, res} -> sec <> res end)
  end

  @doc """
    Loads the training set specified by `path` into memory, generates a new hidden layer
    for a RBF Neural Network, and saves it in the working directory with the name
    `hidden_layer`
  """
  def cluster(path) do
    IO.puts("Reading training set for unsupervised learning...")
    labeled_set = FileUtils.read_csv(path) |> FileUtils.parse_dataset_csv()
    data = Enum.map(labeled_set, fn item -> elem(item,0) end)

    IO.puts("Training Radial Network hidden layer...")
    hidden_layer = RadialNet.init_radial_part(data, 20)

    IO.puts("Exporting hidden layer to file...")
    FileUtils.serialize(hidden_layer, "hidden_layer")

    IO.puts("Done.")
  end

  @doc """
    Loads up the generated hidden layer specified by `hidden_layer_path`, reads
    the training set from disk into memory, initializes a new neural network and
    trains it. The resulting net is then saved to the working directory with
    the name `neural net`
  """
  def train(hidden_layer_path, training_set_path, epochs_string) do
    IO.puts("Reading hidden layer from file.")
    hidden_layer = FileUtils.deserialize(hidden_layer_path)

    IO.puts("Reading training data from file.")
    labeled_training_set = FileUtils.read_csv(training_set_path) |> FileUtils.parse_dataset_csv
    net = RadialNet.init(hidden_layer, 6)
    num_examples = length(labeled_training_set)
    epochs = String.to_integer(epochs_string)

    IO.puts("Training neural network for #{epochs} epochs...")
    trained_net = RadialNet.train_on_dataset(net, labeled_training_set,num_examples, epochs)

    IO.puts("Saving neural network to file...")
    FileUtils.serialize(trained_net, "neural_net")

    IO.puts("Done")
  end

  @doc """
    Loads up a previously saved neural network from disk indicated by `network_path`, then loads the
    testing set to memory and calculates the net's accuracy, informing the user.
  """
  def calculate_accuracy(network_path, testing_set_path) do
    net = FileUtils.deserialize(network_path)
    labeled_testing_set = FileUtils.read_csv(testing_set_path) |> FileUtils.parse_dataset_csv
    results = RadialNet.calculate_accuracy(net,labeled_testing_set)
    IO.puts("Network is #{results.accuracy}% accurate (#{results.hits} out of #{results.example_count} correct guesses)")
  end

  @doc """
    Loads a neural net into memory as well as a single input, and predicts what the output'll be.
  """
  def process(network_path, input_path) do
    net = FileUtils.deserialize(network_path)
    input = FileUtils.read_csv(input_path) |> FileUtils.parse_csv_line
    results = RadialNet.predict(net, input) |> ProgramUtils.interpret
    IO.puts("Resultados de la prueba:")
    Enum.map(results, fn section_result -> IO.puts(section_result) end)
  end

  @doc """
    Displays the program's inline help.
  """
  def display_help() do
    help_string = "usage: rbf [options]
      options:
        --cluster training_set_path      Creates the network's hidden layer and saves it to a file.
        --train hidden_layer_path training_set_path num_epochs    Initializes the net's output layer, trains it, and saves it to a file.
        --accuracy neural_network_path testing_set_path    Calculate a neural network's accuracy.
        --test neural_network_path example_path   Processes a single input.
      note:
        The --train flag depends on the files generated by running the program with the --cluster tag,
        and both the --accuracy and --test flags depend on the files generated by the --train flag.
        Take great care to run the commands in the correct order, lest you run into an error.
    "
    IO.puts(help_string)
  end

end

args = System.argv()

case args do
  ["--cluster", training_set_path] -> ProgramUtils.cluster(training_set_path)
  ["--train", hidden_layer_path,training_set_path, epochs] -> ProgramUtils.train(hidden_layer_path, training_set_path, epochs)
  ["--accuracy", network_path, testing_set_path] -> ProgramUtils.calculate_accuracy(network_path, testing_set_path)
  ["--test", network_path, example_path] -> ProgramUtils.process(network_path, example_path)
  ["--help"] -> ProgramUtils.display_help()
  _ -> IO.puts("Unrecognized command. Try running the script with the --help option flag.")
end
