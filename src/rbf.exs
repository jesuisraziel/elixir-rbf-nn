defmodule FileUtils do
  def read_csv(path) do
    File.stream!(path) |> Enum.to_list
  end

  def parse_csv_line(list) do
    hd(list)
    |> String.replace("\n", "")
    |> String.split(",")
    |> Enum.map(fn elem -> String.to_integer(elem) end)
  end

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
end

defmodule MathUtils do
  def random_float(lower,upper) do
    :rand.uniform() * (lower - upper) + upper
  end

  def euclidean_distance(a,b) when length(a) == length(b) do
    Enum.zip(a,b)
    |> Enum.reduce(0,fn {an, bn}, acc -> :math.pow(bn-an,2)+acc end)
    |> :math.sqrt
  end
end

defmodule NetUtils do
  def radial_factor(input, centroid, deviation)
    when (length(input) == length(centroid))
    and is_number(deviation)
  do
    MathUtils.euclidean_distance(centroid, input)/deviation
  end

  def gaussian(input, centroid, deviation) do
    radius = NetUtils.radial_factor(input, centroid, deviation)
    numerator = -(:math.pow(radius, 2))
    denominator = 2
    :math.exp(numerator/denominator)
  end

  def linear(input) do
    input
  end
end

defmodule VecOps do
  def dot(inputs,weights)
    when length(inputs) == length(weights)
  do
    Enum.zip(inputs, weights) |> Enum.map(fn {a, b} -> a*b end) |> Enum.sum
  end

  def add(a,b)
    when length(a) == length(b)
  do
    Enum.zip(a,b) |> Enum.map(fn {a,b} -> a+b end)
  end

  def scale(a,b)
    when is_list(a)
  do
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
      %{centroids: centroids, clustered_data: clusters}
    end
  end

  defp init_centroids(dataset,k) when is_list(dataset) do
    #Select a random dataset point to act as the centroid
    shape = hd(dataset) |> length
    for _ <- 1..k, do: (
      for _ <- 1..shape, do: MathUtils.random_float(-2,2)
    )
  end

  defp assign_clusters(dataset, centroids) do
    for data <- dataset, do:
    {data,  centroids
      |> Enum.map(fn centroid -> MathUtils.euclidean_distance(centroid, data) end)
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
        Enum.reduce(cluster, initial_accumulator, fn datapoint, acc -> VecOps.add(acc, elem(datapoint,0)) end)
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
      |> Enum.map(fn cen_2 -> MathUtils.euclidean_distance(cen, cen_2) end)
      |> Enum.sort
      |> Enum.take(2)
      |> List.to_tuple
    )
    for min <- cen_mins, do: :math.sqrt(elem(min,0)*elem(min,1))
  end
end

defmodule RadialNeuron do
  defstruct [:id, :center, :deviation]

  def init(id, center, deviation) do
    %RadialNeuron{id: id, center: center, deviation: deviation}
  end

  def activate(neuron, input)  do
    NetUtils.gaussian(input, neuron.center, neuron.deviation)
  end
end

defmodule OutputNeuron do
  defstruct [:id, :weights, :bias]

  def init(id, num_weights) when is_integer(num_weights) do
    w = for _ <- 1..num_weights, do: MathUtils.random_float(-2,2)
    b = MathUtils.random_float(0,1)
    %OutputNeuron{id: id, weights: w, bias: b}
  end

  def activate(neuron, input)  do
    VecOps.dot(neuron.weights, input) |> (Kernel.+neuron.bias) |> NetUtils.linear |> Kernel.round()
  end

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
  defstruct [:radial_neurons, :output_neurons, learning_rate: 0.07]

  def init_radial_part(dataset, cluster_num) do
    clusters = KMeans.run(dataset,cluster_num) #|> IO.inspect
    KNeighbors.run(clusters.centroids)
  end

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

  def activate_hidden_layer(net, input) do
    for neuron <- net.radial_neurons, do: (
      RadialNeuron.activate(neuron, input)
    )
  end

  def predict(net, input) do
    hidden_layer_output = activate_hidden_layer(net, input)
    for neuron <- net.output_neurons, do: OutputNeuron.activate(neuron, hidden_layer_output)
  end

  def mean_square_error(net, dataset, example_count) do
    outputs = for example <- dataset, do: RadialNet.predict(net, elem(example,0))
    labels = Enum.map(dataset, fn example -> elem(example,1) end)
    labeled_outputs = Enum.zip(labels,outputs)
    squared_errors = for {expected_values,actual_values} <- labeled_outputs, do: (
      Enum.zip(expected_values,actual_values)
      |> Enum.map(fn {expected,actual} -> :math.pow(expected-actual,2) end)
      |> Enum.reduce(0, fn squared_value_error, acc -> squared_value_error + acc end)
      |> (Kernel./2)
    )
    Enum.reduce(squared_errors, 0, fn error, acc -> (error+acc) end)
    |> (Kernel./example_count)
  end

  def train_on_dataset(net, labeled_dataset,example_count, epochs) do
    if(epochs == 0) do
      net
    else
      IO.puts("Epoca #{to_string(epochs)}")
      #_mse = RadialNet.mean_square_error(net, labeled_dataset, example_count) |> IO.inspect
      updated_net = train_on_dataset(net, labeled_dataset) #|> IO.inspect
      #_ = IO.gets("Pausing")
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

  def calculate_accuracy(net, labeled_dataset) do
    data = Enum.map(labeled_dataset, fn item -> elem(item,0) end)
    labels = Enum.map(labeled_dataset, fn item -> elem(item,1) end)
    outputs = Enum.map(data, fn datum -> RadialNet.predict(net, datum) end)
    zipped_outputs = Enum.zip(labels,outputs) #|> IO.inspect
    example_count = length(labeled_dataset)
    hits = Enum.reduce(zipped_outputs, 0, fn {exp, act}, acc -> if exp == act do acc + 1 else acc + 0 end end) #|> IO.inspect
    accuracy = hits/length(labeled_dataset)
    %{example_count: example_count, hits: hits, accuracy: accuracy}
  end
end

defmodule DataUtils do
  def separate_labels_and_data(labeled_dataset) do
    data = Enum.map(labeled_dataset, fn item -> elem(item,0) end)
    labels = Enum.map(labeled_dataset, fn item -> elem(item,1) end)
    {data, labels}
  end

  def get_dataset(dataset_path) do
    FileUtils.read_csv(dataset_path) |> FileUtils.parse_dataset_csv
  end
end

defmodule ProgramUtils do

  def serialize(element, path) do
    bin = :erlang.term_to_binary(element)
    File.write!(path, bin)
  end

  def deserialize(path) do
    File.read!(path) |> :erlang.binary_to_term
  end

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

  def cluster(path) do
    IO.puts("Reading training set for unsupervised learning...")
    labeled_set = FileUtils.read_csv(path) |> FileUtils.parse_dataset_csv()
    data = Enum.map(labeled_set, fn item -> elem(item,0) end)

    IO.puts("Training Radial Network hidden layer...")
    hidden_layer = RadialNet.init_radial_part(data, 20)

    IO.puts("Exporting hidden layer to file...")
    serialize(hidden_layer, "hidden_layer")

    IO.puts("Done.")
  end

  def train(hidden_layer_path, training_set_path, epochs_string) do
    IO.puts("Reading hidden layer from file.")
    hidden_layer = ProgramUtils.deserialize(hidden_layer_path)

    IO.puts("Reading training data from file.")
    labeled_training_set = FileUtils.read_csv(training_set_path) |> FileUtils.parse_dataset_csv
    net = RadialNet.init(hidden_layer, 6)
    num_examples = length(labeled_training_set)
    epochs = String.to_integer(epochs_string)

    IO.puts("Training neural network for #{epochs} epochs...")
    trained_net = RadialNet.train_on_dataset(net, labeled_training_set,num_examples, epochs)

    IO.puts("Saving neural network to file...")
    serialize(trained_net, "neural_net")

    IO.puts("Done")
  end

  def calculate_accuracy(network_path, testing_set_path) do
    net = deserialize(network_path)
    labeled_testing_set = FileUtils.read_csv(testing_set_path) |> FileUtils.parse_dataset_csv
    results = RadialNet.calculate_accuracy(net,labeled_testing_set)
    IO.puts("Network is #{results.accuracy}% accurate (#{results.hits} out of #{results.example_count} correct guesses)")
  end

  def process(network_path, input_path) do
    net = deserialize(network_path)
    input = FileUtils.read_csv(input_path) |> FileUtils.parse_csv_line
    results = RadialNet.predict(net, input) |> ProgramUtils.interpret
    IO.puts("Resultados de la prueba:")
    Enum.map(results, fn section_result -> IO.puts(section_result) end)
  end

  def display_help() do
    help_string = "usage: rbf [options]
      options:
        --cluster training_set_path      Creates the network's hidden layer and saves it to a file.
        --train hidden_layer_path training_set_path num_epochs    Initializes the net's output layer, trains it, and saves it to a file.
        --accuracy neural_network_path testing_set_path    Calculate a neural network's accuracy.
        --test neural_network_path example_path   Processes a single input.
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
