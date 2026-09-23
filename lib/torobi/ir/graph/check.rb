# frozen_string_literal: true

module Torobi
  module IR
    class Graph
      # What makes a graph a graph, checked before one is made:
      #
      # - ids are consecutive from 0 within each kind
      # - input names, parameter paths and node names are unique
      # - a node references only inputs and earlier nodes (no forward references)
      # - every node is reachable from an output (no dead nodes)
      #
      # Separate from Graph so that Graph is the value and its queries, and
      # this is the one place the rules are read from.
      class Check
        def initialize(inputs:, parameters:, nodes:)
          @inputs = inputs
          @parameters = parameters
          @nodes = nodes
        end

        # Checks everything and returns +outputs+ as the graph holds them:
        # string names to Refs, sorted, frozen.
        def call(outputs)
          sequential!(@inputs, InputSpec, "input")
          sequential!(@parameters, ParameterSpec, "parameter")
          sequential!(@nodes, NodeSpec, "node")
          unique!(@inputs.map(&:name), "input name")
          unique!(@parameters.map(&:path), "parameter path")
          unique!(@nodes.filter_map(&:name), "node name")
          @nodes.each { |node| references!(node) }
          outputs = outputs!(outputs)
          reachable!(outputs)
          outputs
        end

        private

        def sequential!(specs, klass, kind)
          specs.each_with_index do |spec, i|
            unless spec.is_a?(klass)
              raise ConfigError, "#{kind} at position #{i} is a #{spec.class}, expected #{klass}"
            end
            unless spec.id == i
              raise ConfigError,
                    "#{kind} ids must be consecutive from 0: found id #{spec.id} at position #{i}"
            end
          end
        end

        def unique!(values, what)
          values.tally.each do |value, count|
            raise ConfigError, "duplicate #{what} #{value.inspect}" if count > 1
          end
        end

        # Only earlier nodes are within reach, which is what forbids a
        # forward reference.
        def references!(node)
          where = "node #{node.id} (#{node.op})"
          node.inputs.each do |ref|
            if ref.node? && ref.id >= node.id
              raise ConfigError,
                    "#{where}: references #{ref}, which is not before it " \
                    "(forward references are not allowed)"
            end
            ref.resolve(inputs: @inputs, nodes: @nodes, where: "#{where}:")
          end
          node.parameters.each do |pid|
            unless pid < @parameters.size
              raise ConfigError, "#{where}: references unknown parameter #{pid}"
            end
          end
        end

        def outputs!(outputs)
          unless outputs.is_a?(Hash) && !outputs.empty?
            raise ConfigError, "a graph must declare at least one named output"
          end

          named = outputs.to_h do |name, ref|
            name = name.to_s
            raise ConfigError, "an output name must not be empty" if name.empty?

            ref = Ref.parse(ref)
            ref.resolve(inputs: @inputs, nodes: @nodes, where: "output #{name.inspect}")
            [-name, ref]
          end
          named.sort.to_h.freeze
        end

        # Every node must contribute to an output. A dead node is almost
        # always a mistake in the model definition, and silently keeping it
        # would make the digest depend on code that does nothing.
        def reachable!(outputs)
          alive = []
          queue = node_ids(outputs.values)
          until queue.empty?
            id = queue.pop
            next if alive[id]

            alive[id] = true
            queue.concat(node_ids(@nodes[id].inputs))
          end
          dead = @nodes.reject { |n| alive[n.id] }
          return if dead.empty?

          names = dead.map { |n| "node #{n.id} (#{n.op})" }.join(", ")
          raise ConfigError, "unreachable from any output: #{names}"
        end

        def node_ids(refs) = refs.filter_map { |ref| ref.id if ref.node? }
      end
    end
  end
end
