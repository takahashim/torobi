# frozen_string_literal: true

module Torobi
  module IR
    # One computation graph: inputs, parameters, nodes in topological order,
    # and its named outputs. Construction validates the structure, so an
    # invalid graph is never representable:
    #
    # - ids are consecutive from 0 within each kind
    # - input names and parameter paths are unique
    # - a node references only inputs and earlier nodes (no forward references)
    # - every node is reachable from an output (no dead nodes)
    #
    # Outputs are named because that is what an objective graph refers to
    # (docs/plan.md section 5A.3): {"logits" => "node:12"}.
    class Graph < Data.define(:inputs, :parameters, :nodes, :outputs)
      def initialize(inputs:, parameters:, nodes:, outputs:)
        inputs = sequential!(inputs, InputSpec, "input")
        parameters = sequential!(parameters, ParameterSpec, "parameter")
        nodes = sequential!(nodes, NodeSpec, "node")
        unique!(inputs.map(&:name), "input name")
        unique!(parameters.map(&:path), "parameter path")
        unique!(nodes.filter_map(&:name), "node name")

        nodes.each { |node| check_references(node, inputs, parameters) }
        outputs = check_outputs(outputs, inputs, nodes)
        check_reachability(nodes, outputs)

        super(inputs: Freeze.deep(inputs.dup), parameters: Freeze.deep(parameters.dup),
              nodes: Freeze.deep(nodes.dup), outputs:)
      end

      def to_h
        {
          "inputs" => inputs.map(&:to_h),
          "parameters" => parameters.map(&:to_h),
          "nodes" => nodes.map(&:to_h),
          "outputs" => outputs
        }
      end

      # The shape and dtype behind one named output, for whoever consumes it.
      def output_signature(name)
        ref = outputs.fetch(name.to_s) do
          raise ConfigError,
                "this graph has no output #{name.to_s.inspect}; it has " \
                "#{outputs.keys.map(&:inspect).join(", ")}"
        end
        kind, id = Ref.parse(ref)
        spec = kind == :input ? inputs[id] : nodes[id]
        [spec.shape, spec.dtype]
      end

      def input_named(name)
        inputs.find { |i| i.name == name.to_s }
      end

      # The node a tap asks for, by its stable name (docs/plan.md 6.4).
      def node_named(name)
        nodes.find { |n| n.name == name.to_s }
      end

      # Every name a tap could ask for.
      def node_names = nodes.filter_map(&:name)

      def self.from_h(h)
        new(inputs: h.fetch("inputs").map { |x| InputSpec.from_h(x) },
            parameters: h.fetch("parameters").map { |x| ParameterSpec.from_h(x) },
            nodes: h.fetch("nodes").map { |x| NodeSpec.from_h(x) },
            outputs: h.fetch("outputs"))
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
        specs
      end

      def unique!(values, what)
        values.tally.each do |value, count|
          raise ConfigError, "duplicate #{what} #{value.inspect}" if count > 1
        end
      end

      def check_references(node, inputs, parameters)
        where = "node #{node.id} (#{node.op})"
        node.inputs.each do |ref|
          kind, id = Ref.parse(ref)
          case kind
          when :input
            raise ConfigError, "#{where}: references unknown input:#{id}" unless id < inputs.size
          when :node
            unless id < node.id
              raise ConfigError,
                    "#{where}: references node:#{id}, which is not before it " \
                    "(forward references are not allowed)"
            end
          end
        end
        node.parameters.each do |pid|
          unless pid < parameters.size
            raise ConfigError, "#{where}: references unknown parameter #{pid}"
          end
        end
      end

      def check_outputs(outputs, inputs, nodes)
        unless outputs.is_a?(Hash) && !outputs.empty?
          raise ConfigError, "a graph must declare at least one named output"
        end

        named = outputs.to_h do |name, ref|
          name = name.to_s
          raise ConfigError, "an output name must not be empty" if name.empty?

          kind, id = Ref.parse(ref)
          limit = kind == :input ? inputs.size : nodes.size
          unless id < limit
            raise ConfigError, "output #{name.inspect} references unknown #{kind}:#{id}"
          end

          [-name, -ref.to_s]
        end
        Freeze.deep(named.sort.to_h)
      end

      # Every node must contribute to an output. A dead node is almost always
      # a mistake in the model definition, and silently keeping it would make
      # the digest depend on code that does nothing.
      def check_reachability(nodes, outputs)
        alive = []
        queue = outputs.each_value.filter_map do |ref|
          kind, id = Ref.parse(ref)
          kind == :node ? id : nil
        end
        until queue.empty?
          id = queue.pop
          next if alive[id]

          alive[id] = true
          nodes[id].inputs.each do |ref|
            kind, ref_id = Ref.parse(ref)
            queue << ref_id if kind == :node
          end
        end
        dead = nodes.reject { |n| alive[n.id] }
        return if dead.empty?

        names = dead.map { |n| "node #{n.id} (#{n.op})" }.join(", ")
        raise ConfigError, "unreachable from any output: #{names}"
      end
    end
  end
end
