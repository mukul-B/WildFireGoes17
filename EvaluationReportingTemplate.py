import logging


class EvaluatationReporting_reg:
    def __init__(self):
        self.evals = {
            'control': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'predicted': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'typecount': {}
        }


    def update(self, eval_single):
        
        type = eval_single.type
        self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + eval_single.IOU_control
        self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_control_inter
        self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_control_union
        self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + eval_single.IOU_predicted
        self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_predicted_inter
        self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_predicted_union
        # self.evals['predicted']['avg_acuracy'][type] = self.evals['predicted']['avg_acuracy'].get(type, 0.0) + eval_single.coverage_converage
        self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    
    def get_evaluations(self):
        return self.evals
    
    def report_results(self):
        evaluations = self.evals
        st = ''.join(['-' for _ in range(128)])
        logging.info(st)
        logging.info("| {:<25}| {:<20}| {:<20}| {:<20}| {:<20}| {:<10}|".format("Type", "Control/Predicted", "IOU",
                                                                                "PSNR_intersection", "PSNR_union", "Count"))
        logging.info(st)
        count = sum(evaluations['typecount'].values())
        for type in evaluations['typecount']:
        # for coverage_type in [LC, HC]:
        #     for iou_type in [LI, HI]:
                # type = coverage_type + iou_type
                # type_name = coverage_type + iou_type
                # if type not in evaluations['typecount']:
                #     continue
                type_name = type
                count2 = evaluations['typecount'][type]
                
                avg_IOU_predicted = evaluations['predicted']['avg_IOU']
                avg_IOU_control = evaluations['control']['avg_IOU']
                avg_psnr_predicted_inter = evaluations['predicted']['avg_psnr_inter']
                avg_psnr_control_inter = evaluations['control']['avg_psnr_inter']
                avg_psnr_predicted_union = evaluations['predicted']['avg_psnr_union']
                avg_psnr_control_union = evaluations['control']['avg_psnr_union']

                IOU_predicted = avg_IOU_predicted[type] / count2
                IOU_control = avg_IOU_control[type] / count2
                PSNR_predicted_intersection = avg_psnr_predicted_inter[type] / count2
                PSNR_control_intersection = avg_psnr_control_inter[type] / count2
                PSNR_predicted_union = avg_psnr_predicted_union[type] / count2
                PSNR_control_union = avg_psnr_control_union[type] / count2

                logging.info("| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Control", IOU_control,
                                                                                        PSNR_control_intersection,
                                                                                        PSNR_control_union, count2))
                logging.info(
                    "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Predicted", IOU_predicted,
                                                                                PSNR_predicted_intersection,
                                                                                PSNR_predicted_union, count2))

        logging.info(st)

        logging.info(
            "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Control",
                                                                        sum(avg_IOU_control.values()) / count,
                                                                        sum(avg_psnr_control_inter.values()) / count,
                                                                        sum(avg_psnr_control_union.values()) / count,
                                                                        count))
        logging.info(
            "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Predicted",
                                                                        sum(avg_IOU_predicted.values()) / count,
                                                                        sum(avg_psnr_predicted_inter.values()) / count,
                                                                        sum(avg_psnr_predicted_union.values()) / count,
                                                                        count))
        logging.info(st)

class EvaluatationReporting_Seg:
    def __init__(self):
        self.evals = {
            'control': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'predicted': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {}
            },
            'typecount': {}
        }

    # def update(self, type, IOU_control, psnr_control_inter, psnr_control_union, IOU_predicted, psnr_predicted_inter, psnr_predicted_union):
    #     self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + IOU_control
    #     self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + psnr_control_inter
    #     self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + psnr_control_union
    #     self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + IOU_predicted
    #     self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + psnr_predicted_inter
    #     self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + psnr_predicted_union
    #     self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    def update(self, eval_single):
        
        type = eval_single.type
        self.evals['control']['avg_IOU'][type] = self.evals['control']['avg_IOU'].get(type, 0.0) + eval_single.IOU_control
        self.evals['control']['avg_psnr_inter'][type] = self.evals['control']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_control_inter
        self.evals['control']['avg_psnr_union'][type] = self.evals['control']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_control_union
        self.evals['predicted']['avg_IOU'][type] = self.evals['predicted']['avg_IOU'].get(type, 0.0) + eval_single.IOU_predicted
        self.evals['predicted']['avg_psnr_inter'][type] = self.evals['predicted']['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_predicted_inter
        self.evals['predicted']['avg_psnr_union'][type] = self.evals['predicted']['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_predicted_union
        # self.evals['predicted']['avg_acuracy'][type] = self.evals['predicted']['avg_acuracy'].get(type, 0.0) + eval_single.coverage_converage
        self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    
    def get_evaluations(self):
        return self.evals
    
    def report_results(self):
        evaluations = self.evals
        st = ''.join(['-' for _ in range(128)])
        logging.info(st)
        logging.info("| {:<25}| {:<20}| {:<20}| {:<20}| {:<20}| {:<10}|".format("Type", "Control/Predicted", "IOU",
                                                                                "PSNR_intersection", "PSNR_union", "Count"))
        logging.info(st)
        count = sum(evaluations['typecount'].values())
        for type in evaluations['typecount']:
        # for coverage_type in [LC, HC]:
        #     for iou_type in [LI, HI]:
                # type = coverage_type + iou_type
                # type_name = coverage_type + iou_type
                # if type not in evaluations['typecount']:
                #     continue
                type_name = type
                count2 = evaluations['typecount'][type]
                
                avg_IOU_predicted = evaluations['predicted']['avg_IOU']
                avg_IOU_control = evaluations['control']['avg_IOU']
                avg_psnr_predicted_inter = evaluations['predicted']['avg_psnr_inter']
                avg_psnr_control_inter = evaluations['control']['avg_psnr_inter']
                avg_psnr_predicted_union = evaluations['predicted']['avg_psnr_union']
                avg_psnr_control_union = evaluations['control']['avg_psnr_union']

                IOU_predicted = avg_IOU_predicted[type] / count2
                IOU_control = avg_IOU_control[type] / count2
                PSNR_predicted_intersection = avg_psnr_predicted_inter[type] / count2
                PSNR_control_intersection = avg_psnr_control_inter[type] / count2
                PSNR_predicted_union = avg_psnr_predicted_union[type] / count2
                PSNR_control_union = avg_psnr_control_union[type] / count2

                logging.info("| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Control", IOU_control,
                                                                                        PSNR_control_intersection,
                                                                                        PSNR_control_union, count2))
                logging.info(
                    "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format(type_name, "Predicted", IOU_predicted,
                                                                                PSNR_predicted_intersection,
                                                                                PSNR_predicted_union, count2))

        logging.info(st)

        logging.info(
            "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Control",
                                                                        sum(avg_IOU_control.values()) / count,
                                                                        sum(avg_psnr_control_inter.values()) / count,
                                                                        sum(avg_psnr_control_union.values()) / count,
                                                                        count))
        logging.info(
            "| {:<25}| {:<20}| {:<20} | {:<20}| {:<20}| {:<10}|".format("Total", "Predicted",
                                                                        sum(avg_IOU_predicted.values()) / count,
                                                                        sum(avg_psnr_predicted_inter.values()) / count,
                                                                        sum(avg_psnr_predicted_union.values()) / count,
                                                                        count))
        logging.info(st)

