## 1. llvm::StringRef

​	`llvm::StringRef` 的核心本质就是一个**“视图（View）”**或**“观察者”**，它自己不拥有（不管理）背后的数据，只是单纯地记录了“数据在哪儿（指针）”以及“数据有多长（长度）”。StringRef 对象本身通常分配在栈（Stack）上，但是指向的内容可能在heap也可能是全局或者常量，当然也可是stack，只要保证StringRef 对象的生命周期在指向对象的生命周期以内就可以。



1. LLVM中所有的对象都来自于某种基类，一共有三个体系`llvm::Value` 体系、`llvm::Type` 体系、`mlir::Operation` 体系。比如：

    `llvm::Value` 中的基类里面存了一个枚举值（通常叫 `SubclassID` 或 `Kind`）。

   ```c++
   class Value {
   public:
     // 定义在这个体系下的所有子类枚举
     enum ValueTy {
       ArgumentVal,
       BasicBlockVal,
       FunctionVal,
       InstructionVal,
       // ...
     };
   private:
     const unsigned char SubclassID; // 存储当前对象的真实类型 ID
   
   public:
     unsigned getValueID() const { return SubclassID; }
   };
   ```

   

