const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
        PageBreak, LevelFormat } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

// Helper function to create styled paragraph
function styledParagraph(text, fontSize, bold) {
    return new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: text, font: "SimSun", size: fontSize, bold: bold })]
    });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "SimSun", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 400, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 300, after: 180 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "SimHei" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u25CF", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // Title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        children: [new TextRun({ text: "人工智能基础及应用", font: "SimHei", size: 56, bold: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        children: [new TextRun({ text: "大作业一报告", font: "SimHei", size: 48, bold: true })]
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // Section 1
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("一、问题背景描述")] }),
      new Paragraph({ children: [new TextRun({ text: "1.1 比赛简介", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("IEEE-CIS Fraud Detection是由IEEE计算智能协会举办的欺诈检测竞赛，目标是预测在线交易是否为欺诈行为。该竞赛在Kaggle平台上进行，吸引了来自全球的数据科学家和机器学习工程师参与。", 24, false),
      styledParagraph("比赛链接：https://www.kaggle.com/c/ieee-fraud-detection", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "1.2 问题定义", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("本问题是一个二分类任务，给定交易数据，预测交易是否为欺诈（isFraud=0或isFraud=1）。评估指标为ROC曲线下面积（AUC），该指标能够有效衡量分类器在不同阈值下的整体表现。", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "1.3 数据描述", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("数据集包含两个主表：", 24, false),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("train_transaction.csv：训练集交易数据，包含约59万条记录、394个特征")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("train_identity.csv：训练集身份信息，包含设备相关特征")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("test_transaction.csv / test_identity.csv：测试集数据")] }),
      new Paragraph({ spacing: { after: 240 }, children: [new TextRun("")] }),
      styledParagraph("数据特点：", 24, false),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("高度类别不平衡：欺诈交易约占3.5%")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("特征维度高：包含V、C、D、M等系列特征")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("存在大量缺失值：部分特征缺失率超过50%")] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 2
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("二、数据探索与预处理")] }),
      new Paragraph({ children: [new TextRun({ text: "2.1 数据加载与合并", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("通过TransactionID将交易表与身份表进行左连接合并。由于身份信息仅存在于部分交易中，合并后存在较多缺失值。", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "2.2 缺失值处理", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("（1）删除高缺失率特征：删除缺失率超过50%的列。", 24, false),
      styledParagraph("（2）数值特征填充：使用-999填充数值列的缺失值。", 24, false),
      styledParagraph("（3）类别特征填充：使用Unknown填充类别列的缺失值。", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "2.3 类别特征编码", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("使用Label Encoding将类别特征转换为数值形式，便于模型处理。", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "2.4 PCA降维与可视化", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("对C特征（C1-C14）进行PCA降维，保留10个主成分：", 24, false),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3000, 2000, 4026],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "C特征PCA", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "参数", bold: true })] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "数值", bold: true })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("原始特征数")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("C1-C14")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("14个")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("PCA主成分数")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("n_components")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("10")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("解释方差比")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("explained_variance_ratio")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.9998")] })] }),
          ]}),
        ]
      }),

      styledParagraph("PCA可视化结果说明：通过对V特征、C特征、D特征等进行PCA和t-SNE降维可视化，观察到欺诈样本和非欺诈样本在低维空间中高度重叠。在进行PCA后，删除了原始的C特征，仅保留PCA降维后的10个主成分。", 24, false),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 3
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("三、特征工程")] }),
      new Paragraph({ children: [new TextRun({ text: "3.1 时间特征提取", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("从TransactionDT特征中提取时间相关特征：", 24, false),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("hour：小时（0-23）")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("day：一周中的第几天（0-6）")] }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "3.2 金额特征处理", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("对TransactionAmt进行特征变换：", 24, false),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("TransactionAmt_log：对数变换，log(1+x)")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("TransactionAmt_cents：提取小数部分")] }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "3.3 最终特征组合", font: "SimHei", bold: true, size: 28 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3000, 3026, 3000],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "特征类别", bold: true })] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "数量", bold: true })] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "说明", bold: true })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("V特征")] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("V1-V339")] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("交易相关匿名特征")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("C特征PCA")] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("10")] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("C1-C14降维后")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("D特征")] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("D1-D15")] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("时间相关特征")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("Card/Addr特征")] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("约15")] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("卡片和地址信息")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "E0E0E0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "合计", bold: true })] })] }),
            new TableCell({ borders, width: { size: 3026, type: WidthType.DXA }, shading: { fill: "E0E0E0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "218", bold: true })] })] }),
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "E0E0E0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "最终使用", bold: true })] })] }),
          ]}),
        ]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 4
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("四、模型对比")] }),
      new Paragraph({ children: [new TextRun({ text: "4.1 模型选择", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("根据大作业要求，我们选择以下三种模型进行对比：", 24, false),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("逻辑回归（Logistic Regression）：经典的线性分类器，作为基线模型")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("随机森林（Random Forest）：集成学习方法，擅长处理高维数据")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("CatBoost：梯度提升算法，对类别特征和缺失值处理优秀")] }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "4.2 交叉验证策略", font: "SimHei", bold: true, size: 28 })] }),
      styledParagraph("采用5折分层交叉验证（StratifiedKFold），保证每折中欺诈样本的比例与原始数据一致。", 24, false),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "4.3 模型训练结果", font: "SimHei", bold: true, size: 28 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3000, 2000, 2000, 2026],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "模型", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "OOF AUC", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Fold 1", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "Fold 2", bold: true })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("Logistic Regression")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8170")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8185")] })] }),
            new TableCell({ borders, width: { size: 2026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8216")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("Random Forest")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8917")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8931")] })] }),
            new TableCell({ borders, width: { size: 2026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8926")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "CatBoost (最佳)", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "0.9098", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "0.9128", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2026, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, children: [new Paragraph({ children: [new TextRun({ text: "0.9127", bold: true })] })] }),
          ]}),
        ]
      }),

      styledParagraph("结果分析：CatBoost模型表现最佳，OOF AUC达到0.9098，明显优于其他两个模型。这是因为CatBoost采用了Ordered Boosting技术，能够有效减少预测偏移。", 24, false),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 5
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("五、调参与评估")] }),
      new Paragraph({ children: [new TextRun({ text: "5.1 CatBoost参数设置", font: "SimHei", bold: true, size: 28 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3000, 2000, 4026],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "参数", bold: true })] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "值", bold: true })] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "说明", bold: true })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("iterations")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("1000")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("最大迭代次数")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("learning_rate")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.03")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("较低学习率")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("depth")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("6")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("树深度，防止过拟合")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("l2_leaf_reg")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("10")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("L2正则化")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("early_stopping_rounds")] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("100")] })] }),
            new TableCell({ borders, width: { size: 4026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("早停轮数")] })] }),
          ]}),
        ]
      }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "5.2 防止过拟合策略", font: "SimHei", bold: true, size: 28 })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("增加L2正则化（l2_leaf_reg=10）")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("控制树深度（depth=6）")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("使用早停机制（early_stopping_rounds=100）")] }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "5.3 最终评估指标", font: "SimHei", bold: true, size: 28 })] }),

      new Table({
        width: { size: 9026, type: WidthType.DXA },
        columnWidths: [3000, 6026],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "指标", bold: true })] })] }),
            new TableCell({ borders, width: { size: 6026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun({ text: "数值", bold: true })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("交叉验证 AUC")] })] }),
            new TableCell({ borders, width: { size: 6026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.9098")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("Kaggle Private Score")] })] }),
            new TableCell({ borders, width: { size: 6026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("0.8933")] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("Kaggle 排名")] })] }),
            new TableCell({ borders, width: { size: 6026, type: WidthType.DXA }, children: [new Paragraph({ children: [new TextRun("4621 / 约6600")] })] }),
          ]}),
        ]
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 6
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("六、总结")] }),
      styledParagraph("本项目完成了IEEE-CIS欺诈检测竞赛的数据分析与建模工作，主要成果包括：", 24, false),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("数据预处理：合并交易表与身份表，处理高缺失率特征，填充缺失值")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("特征工程：提取时间特征，处理金额特征，对C特征进行PCA降维")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("模型对比：训练并对比逻辑回归、随机森林和CatBoost三种模型")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("调参与评估：通过正则化、早停等策略防止过拟合，最终获得Private AUC 0.8933")] }),

      new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: "改进方向：", bold: true })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("特征工程：添加更多聚合特征")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("模型融合：结合LightGBM、XGBoost等多模型进行集成学习")] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun("超参数优化：使用Optuna等工具进行系统性超参数搜索")] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Section 7
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("七、参考资料")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("Kaggle IEEE-CIS Fraud Detection比赛主页：https://www.kaggle.com/c/ieee-fraud-detection")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("CatBoost官方文档：https://catboost.ai/en/docs/")] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: [new TextRun("Scikit-learn文档：https://scikit-learn.org/stable/")] }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("D:/玛卡巴卡/MATERIALS/22spring/人工智能基础及应用/大作业一报告.docx", buffer);
  console.log("Report generated successfully!");
}).catch(err => {
  console.error("Error:", err);
});