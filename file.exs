defmodule FileUtils do
def read_csv(path) do
  File.stream!(path) |> Enum.to_list
end

def parse_csv(list) when list == [] do
  list
end

def parse_csv(list) do
  [input, label | rest] = list
  parsed_input = input 
                 |> String.replace("\n", "")                
                 |> String.split(",")
                 |> Enum.map(fn elem -> String.to_integer(elem) end)
  parsed_label = label 
                 |> String.replace("\n", "")                
                 |> String.split(",")
                 |> Enum.map(fn elem -> String.to_integer(elem) end)

  [{parsed_input, parsed_label}|parse_csv(rest)]
end
end

FileUtils.read_csv("dataset.csv") 
|> FileUtils.parse_csv 
|> IO.inspect |> length |> IO.inspect
