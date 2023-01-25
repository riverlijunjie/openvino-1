# OpenVINO™ CPU plugin

## Common flow

```mermaid
graph TB
    subgraph CPUPlugin [CPU Plugin]
        subgraph OVModel [OV model level]
            OVGraph[OV Graph]
            OVOpset[OV Opset]
            CPUOpset[CPU Opset]
            CPUTransformationPipeline[CPU transformation pipeline]
            OVGraph -.- OVOpset
            OVOpset --> CPUTransformationPipeline
            CPUTransformationPipeline --> CPUOpset

            %% style OVModel fill:#86BDFF,stroke:#333,stroke-width:2px
        end

        subgraph InternalGraph [Internal graph level]
            Graph[Internal Graph]
            Opset[Internal Opset]
            GraphOptimizer[Graph Optimizer]
            Opset --> GraphOptimizer

            Graph -.- Opset

            %% style InternalGraph fill:#86BDFF,stroke:#333,stroke-width:2px
        end

        subgraph ImplementationLevel [Implementation level]
            Executor[Executor]
            Reference[Reference impls]
            Native[Native impls]
            Backend[Thirdparty impls]
            Snippets[Snippets]

            Executor --> Reference
            Executor --> Native
            Executor --> Backend
            Executor --> Snippets

            %% style ImplementationLevel fill:#86BDFF,stroke:#333,stroke-width:2px
        end

        subgraph Thirdparty [Thirdparty level]
            OneDNN[OneDNN]
            ACL[ACL]

            %% style Thirdparty fill:#86BDFF,stroke:#333,stroke-width:2px
        end


        OVModel -->InternalGraph
        InternalGraph --> ImplementationLevel

    end
```


## CPU Transformation Pipeline

```mermaid

graph LR
    subgraph CPUTransformationPipeline [CPU Transformation Pipeline]
       CommonTransformations[Common transformations]
       LPTransformations[LP transformations]
       SnippetsTokenization[Snippets tokenization]
       CPUOpsetConversion[CPU opset conversion]

       CommonTransformations --> LPTransformations
       LPTransformations --> SnippetsTokenization
       SnippetsTokenization --> CPUOpsetConversion
    end
```


```
transformations
├── transformation_pipeline.cpp
├── cpu_opset
│   ├── common
│   │   ├── op
│   │   ├── pass
│   ├── x64
│   │   ├── ...
│   ├── arm
│   │   ├── ...
├── snippets
│   ├── x64
│   │   ├── ...
```

## Internal graph: post ops fusing

```mermaid
graph
    subgraph PostOpsExample [Post ops fusing]
        subgraph Original [Original subgraph]
            Convolution[Convolution]
            Add[Add]
            Relu[Relu]
            FQ[FakeQuantize]

            Convolution --> Add
            Add --> Relu
            Relu --> FQ
        end

        subgraph Fused [Fused subgraph]
            ConvolutionFused[Convolution <br> +fusedWith: Add,Relu,FQ]
        end

        Original --GraphOptimizer --> Fused
    end
```

```
transformations
├── graph.cpp
├── cpu_opset
│   ├── common
│   │   ├── op
│   │   ├── pass
│   ├── x64
│   │   ├── ...
│   ├── arm
│   │   ├── ...
├── snippets
│   ├── x64
│   │   ├── ...
```

## Implementation level: Executor hierarchy

```mermaid
---
title: Executor hierarchy
---
classDiagram
    class Node{
         <<Abstract>>
    }

    class ExecutorFactory{
        <<Abstract>>
    }
    class OpExecutorFactory{
        + makeExecutor(...) opExecutorPtr

        - supportedExecutors : opExecutorBuilderVec
    }

    class OpExecutorBuilder{
        <<Interface>>
        + isSupported(...) bool
        + makeExecutor(...) opExecutorPtr
    }
    class DnnlOpExecutorBuilder{
    }
    class ACLOpExecutorBuilder{
    }
    class RefOpExecutorBuilder{
    }

    class OpExecutor{
        <<Interface>>
        + init(...) bool
        + exec(...) void
        + getImplType()
    }
    class DnnlOpExecutor{
    }
    class ACLOpExecutor{
    }
    class RefOpExecutor{
    }

    Node *-- ExecutorFactory
    ExecutorFactory <|-- OpExecutorFactory

    OpExecutorBuilder <|-- DnnlOpExecutorBuilder
    OpExecutorBuilder <|-- ACLOpExecutorBuilder
    OpExecutorBuilder <|-- RefOpExecutorBuilder

    OpExecutor <|-- DnnlOpExecutor
    OpExecutor <|-- ACLOpExecutor
    OpExecutor <|-- RefOpExecutor

    OpExecutorFactory ..> OpExecutorBuilder
    OpExecutorBuilder ..> OpExecutor
```