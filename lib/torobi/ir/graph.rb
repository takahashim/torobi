# frozen_string_literal: true

module Torobi
  module IR
    # One computation graph: inputs, parameters, nodes in topological order,
    # and the references that are its outputs. Construction validates the
    # structure, so an invalid graph is never representable:
    #
    # - ids are consecutive from 0 within each kind
    # - input names and parameter paths are unique
    # - a node references only inputs and earlier nodes (no forward references)
    # - every node is reachable from an output (no dead nodes)
    class Graph < Data.define(:inputs, :parameters, :nodes, :outputs)
      def initialize(inputs:, parameters:, nodes:, outputs:)
        inputs = sequential!(inputs, InputSpec, "input")
        parameters = sequential!(parameters, ParameterSpec, "parameter")
        nodes = sequential!(nodes, NodeSpec, "node")
        unique!(inputs.map(&:name), "input name")
        unique!(parameters.map(&:path), "parameter path")

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
            unless id < inputs.size
              raise ConfigError, "#{where}: references unknown input:#{id}"
            end
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
        if !outputs.is_a?(Array) || outputs.empty?
          raise ConfigError, "a graph must declare at least one output"
        end

        outputs = outputs.map { |ref| -ref.to_s }
        outputs.each do |ref|
          kind, id = Ref.parse(ref)
          limit = kind == :input ? inputs.size : nodes.size
          unless id < limit
            raise ConfigError, "output references unknown #{kind}:#{id}"
          end
        end
        Freeze.deep(outputs)
      end

      # Every node must contribute to an output. A dead node is almost always
      # a mistake in the model definition, and silently keeping it would make
      # the digest depend on code that does nothing.
      def check_reachability(nodes, outputs)
        alive = []
        queue = outputs.filter_map { |ref| (kind, id = Ref.parse(ref); kind == :node ? id : nil) }
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
